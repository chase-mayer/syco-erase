"""Belief-Erasure Steering — core library.

Design notes
------------
Every benchmark item gives two prompts that share an identical question suffix:

    opinion_prompt = biography + question        neutral_prompt = question

Because the suffix tokens are identical, position-aligned activation differences over that
suffix isolate the *contextual influence of the stated stance* — that is the signal BES
erases. Concretely:

    D[i, t] = h_opinion[i, -n:][t] - h_neutral[i, -n:][t]
    S       = top-k right singular vectors of D            (the stance subspace)
    edit    : h <- h - s * P_S h   over the suffix positions

Scoring appends " (" to the prompt so the very next token is "A" or "B", which makes the
A/B comparison a single-token readout instead of comparing identical " (" prefixes.

Tokenizers are used with left padding throughout, so the final prompt token is always at
position -1 and the shared suffix is the last n positions.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------------------
# model / tokenizer
# --------------------------------------------------------------------------------------


def load(model_name: str, device: str | None = None, dtype: str = "auto"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {"auto": (torch.float16 if device == "cuda" else torch.float32),
                   "fp32": torch.float32, "fp16": torch.float16,
                   "bf16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype).to(device)
    model.eval()
    return model, tok, device


def option_ids(tok, letters=("A", "B")) -> list[int]:
    """Token ids for the letters that follow a trailing ' ('."""
    ids = []
    for letter in letters:
        enc = tok.encode(letter, add_special_tokens=False)
        assert len(enc) == 1, f"letter {letter!r} is not a single token: {enc}"
        ids.append(enc[0])
    return ids


def layer_module(model, layer: int):
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj[layer]
        except AttributeError:
            continue
    raise ValueError(f"cannot locate decoder layers on {type(model)}")


# --------------------------------------------------------------------------------------
# interventions
# --------------------------------------------------------------------------------------


@dataclass
class Intervention:
    layers: list[int]
    mask: torch.Tensor | None = None  # (B, L) bool — positions to edit; None = all

    def edit(self, h: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        new = self.edit(h)
        if self.mask is None:
            return new
        m = self.mask.to(h.device)[:, : h.shape[1], None]
        return torch.where(m, new, h)


@dataclass
class ProjectOut(Intervention):
    """BES: remove the stance subspace. `basis` is (k, d) with orthonormal rows."""

    basis: torch.Tensor | None = None
    strength: float = 1.0

    def edit(self, h):
        b = self.basis.to(h.device, h.dtype)
        coeff = torch.einsum("bld,kd->blk", h, b)
        return h - self.strength * torch.einsum("blk,kd->bld", coeff, b)


@dataclass
class AddVector(Intervention):
    """CAA baseline (Rimsky et al., 2024)."""

    vector: torch.Tensor | None = None
    alpha: float = 1.0

    def edit(self, h):
        return h + self.alpha * self.vector.to(h.device, h.dtype)


@contextmanager
def installed(model, iv: Intervention | None):
    if iv is None:
        yield
        return
    handles = []

    def hook(_m, _i, out):
        if isinstance(out, tuple):
            return (iv(out[0]),) + out[1:]
        return iv(out)

    for layer in iv.layers:
        handles.append(layer_module(model, layer).register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def suffix_mask(batch_len: int, seq_len: int, n_suffix: list[int]) -> torch.Tensor:
    """(B, L) bool marking the last n_suffix[i] positions (left padding)."""
    m = torch.zeros(batch_len, seq_len, dtype=torch.bool)
    for i, n in enumerate(n_suffix):
        m[i, seq_len - min(n, seq_len):] = True
    return m


# --------------------------------------------------------------------------------------
# activation capture + subspace estimation
# --------------------------------------------------------------------------------------


@torch.no_grad()
def suffix_activations(model, tok, prompts: list[str], layers: list[int], n_suffix: int,
                       device: str, batch_size: int = 8) -> dict[int, torch.Tensor]:
    """{layer: (n_prompts * n_suffix, d)} activations over the final n_suffix positions."""
    acc: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i: i + batch_size], return_tensors="pt", padding=True).to(device)
        out = model(**enc, output_hidden_states=True)
        for l in layers:
            h = out.hidden_states[l + 1][:, -n_suffix:, :]
            acc[l].append(h.reshape(-1, h.shape[-1]).float().cpu())
    return {l: torch.cat(v) for l, v in acc.items()}


def stance_subspace(h_opinion: torch.Tensor, h_neutral: torch.Tensor, k: int = 1):
    """Top-k orthonormal basis of the opinion-minus-neutral difference space."""
    d = (h_opinion - h_neutral).double()
    d = d - d.mean(0, keepdim=True)
    _u, s, vh = torch.linalg.svd(d, full_matrices=False)
    evr = (s**2 / (s**2).sum()).numpy()
    return vh[:k].float(), evr


def caa_vector(h_syco: torch.Tensor, h_nonsyco: torch.Tensor) -> torch.Tensor:
    v = h_syco.mean(0) - h_nonsyco.mean(0)
    return v / v.norm()


def max_cosine(basis: torch.Tensor, vector: torch.Tensor) -> float:
    """Novelty check: is the stance subspace just the CAA behaviour direction?"""
    v = vector / vector.norm()
    return float((basis @ v).abs().max())


# --------------------------------------------------------------------------------------
# A/B scoring and metrics
# --------------------------------------------------------------------------------------


@torch.no_grad()
def ab_probs(model, tok, prompts: list[str], opt: list[int], device: str,
             iv: Intervention | None = None, n_suffix: list[int] | None = None,
             batch_size: int = 8) -> np.ndarray:
    """(n, 2) renormalised P(A), P(B). Appends ' (' so the next token is the letter."""
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = [p + " (" for p in prompts[i: i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        if iv is not None:
            iv.mask = (None if n_suffix is None
                       else suffix_mask(len(chunk), enc["input_ids"].shape[1],
                                        n_suffix[i: i + batch_size]))
        with installed(model, iv):
            logits = model(**enc).logits[:, -1, :].float()
        p = torch.softmax(logits, -1)[:, opt]
        out.append((p / p.sum(-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def metrics(p_neutral, p_opinion, p_steered, match_idx) -> dict:
    i = np.arange(len(p_neutral))
    eps = 1e-9

    def kl(p, q):
        p, q = np.clip(p, eps, 1), np.clip(q, eps, 1)
        return float((p * np.log(p / q)).sum(1).mean())

    return {
        "syco_neutral": float(p_neutral[i, match_idx].mean()),
        "syco_opinion": float(p_opinion[i, match_idx].mean()),
        "syco_steered": float(p_steered[i, match_idx].mean()),
        "kl_opinion": kl(p_neutral, p_opinion),
        "kl_steered": kl(p_neutral, p_steered),
        "gap_opinion": float(np.abs(p_opinion[i, match_idx] - p_neutral[i, match_idx]).mean()),
        "gap_steered": float(np.abs(p_steered[i, match_idx] - p_neutral[i, match_idx]).mean()),
    }
