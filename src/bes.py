"""Belief-Erasure Steering — core library.

Every benchmark item gives two prompts that share an identical question suffix:

    opinion_prompt = biography + question        neutral_prompt = question

Position-aligned activation differences over that shared suffix isolate the contextual
influence of the stated stance. BES fits a direction to those differences and projects it
out of the residual stream at inference:

    D[i, t] = h_opinion[i, -n:][t] - h_neutral[i, -n:][t]
    u       = top right singular vector of D - mean(D)     (the stance direction)
    edit    : h <- h - s * (u . h) u

Scoring: every prompt is formatted to end in "(" so the next token is an answer letter, and
probabilities are renormalised over the letters that item actually offers (2 to 7).

Spans are given per item as (start, stop) counted from the end of the sequence, which is
robust to left padding: positions [L - start, L - stop) are edited. None edits everything.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = tuple("ABCDEFGHIJKLM")

# --------------------------------------------------------------------------------------
# model, tokenizer, prompt formatting
# --------------------------------------------------------------------------------------


def load(model_name: str, device: str | None = None, dtype: str = "auto"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype == "auto":
        if device == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    return model, tok, device


def option_ids(tok, letters=LETTERS) -> list[int]:
    """Token id of each letter as produced right after '(' in context.

    Taken from the tokenisation of 'Answer: (' + letter, which handles byte-level BPE (Qwen)
    and SentencePiece (Llama, where a bare 'A' would encode as '▁A'). Asserts the letter
    is a single token that does not merge with '('."""
    head = tok.encode("Answer: (", add_special_tokens=False)
    ids = []
    for letter in letters:
        full = tok.encode("Answer: (" + letter, add_special_tokens=False)
        assert full[: len(head)] == head and len(full) == len(head) + 1, (
            f"'(' + {letter!r} does not tokenise as '(' followed by one letter token: {full}")
        ids.append(full[-1])
    return ids


def format_prompt(tok, text: str, chat: bool = False, system: str | None = None) -> str:
    """Raw: text + ' ('.  Chat: text as the user turn, assistant turn prefilled with '('."""
    if not chat:
        prefix = f"{system.strip()}\n\n" if system else ""
        return prefix + text.rstrip() + " ("
    body = text.rstrip()
    if body.endswith("Answer:"):
        body = body[: -len("Answer:")].rstrip()
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": body}]
    out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "("
    if tok.bos_token and out.startswith(tok.bos_token):
        out = out[len(tok.bos_token):]
    return out


def common_suffix_len(tok, a: str, b: str) -> int:
    ta, tb = tok.encode(a), tok.encode(b)
    n = 0
    while n < min(len(ta), len(tb)) and ta[-1 - n] == tb[-1 - n]:
        n += 1
    return n


def common_prefix_len(tok, a: str, b: str) -> int:
    ta, tb = tok.encode(a), tok.encode(b)
    n = 0
    while n < min(len(ta), len(tb)) and ta[n] == tb[n]:
        n += 1
    return n


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


def _last_logits(model, enc: dict) -> torch.Tensor:
    """Logits at the final position only (avoids materialising the full vocab x length tensor)."""
    try:
        return model(**enc, logits_to_keep=1).logits[:, -1, :]
    except TypeError:
        return model(**enc).logits[:, -1, :]


def _encode(tok, prompts: list[str], device: str) -> dict:
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "position_ids": pos}


# --------------------------------------------------------------------------------------
# interventions
# --------------------------------------------------------------------------------------


@dataclass
class Intervention:
    layers: list[int]
    mask: torch.Tensor | None = None

    def edit(self, h: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        new = self.edit(h)
        if self.mask is None:
            return new
        m = self.mask.to(h.device)[:, -h.shape[1]:, None]
        return torch.where(m, new, h)


@dataclass
class ProjectOut(Intervention):
    """BES: h <- h - s * B^T B h, where `basis` is (k, d) with orthonormal rows."""

    basis: torch.Tensor | None = None
    strength: float = 1.0

    def edit(self, h):
        x = h.float()
        b = self.basis.to(h.device, torch.float32)
        coeff = torch.einsum("bld,kd->blk", x, b)
        return (x - self.strength * torch.einsum("blk,kd->bld", coeff, b)).to(h.dtype)


@dataclass
class AddVector(Intervention):
    """CAA (Rimsky et al., 2024): h <- h + alpha * v."""

    vector: torch.Tensor | None = None
    alpha: float = 1.0

    def edit(self, h):
        return h + self.alpha * self.vector.to(h.device, h.dtype)


@dataclass
class AffineErase(Intervention):
    """LEACE (Belrose et al., 2023): h <- h - s * (h - mu) @ R^T @ L^T."""

    proj_left: torch.Tensor | None = None
    proj_right: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    strength: float = 1.0

    def edit(self, h):
        dt = torch.float32
        x = h.to(dt)
        delta = x - self.bias.to(h.device, dt)
        corr = (delta @ self.proj_right.to(h.device, dt).T) @ self.proj_left.to(h.device, dt).T
        return (x - self.strength * corr).to(h.dtype)


@contextmanager
def installed(model, iv: Intervention | None):
    if iv is None:
        yield
        return

    def hook(_m, _i, out):
        if isinstance(out, tuple):
            return (iv(out[0]),) + tuple(out[1:])
        return iv(out)

    handles = [layer_module(model, layer).register_forward_hook(hook) for layer in iv.layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def span_mask(seq_len: int, spans: list[tuple[int, int]]) -> torch.Tensor:
    """(B, L) bool; row i marks positions [L - start_i, L - stop_i)."""
    m = torch.zeros(len(spans), seq_len, dtype=torch.bool)
    for i, (start, stop) in enumerate(spans):
        lo, hi = seq_len - min(start, seq_len), seq_len - min(stop, seq_len)
        m[i, lo:hi] = True
    return m


def suffix_mask(batch_len: int, seq_len: int, n_suffix: list[int]) -> torch.Tensor:
    """(B, L) bool marking the last n_suffix[i] positions."""
    assert batch_len == len(n_suffix)
    return span_mask(seq_len, [(n, 0) for n in n_suffix])


# --------------------------------------------------------------------------------------
# activation capture + direction estimation
# --------------------------------------------------------------------------------------


@torch.no_grad()
def suffix_activations(model, tok, prompts: list[str], layers: list[int], n_suffix: int,
                       device: str, batch_size: int = 8) -> dict[int, torch.Tensor]:
    """{layer: (n_prompts * n_suffix, d)} decoder-layer outputs over the final n positions."""
    acc: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    store: dict[int, torch.Tensor] = {}

    def make_hook(l):
        def hook(_m, _i, out):
            store[l] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    backbone = getattr(model, "model", model)
    handles = [layer_module(model, l).register_forward_hook(make_hook(l)) for l in layers]
    try:
        for i in range(0, len(prompts), batch_size):
            enc = _encode(tok, prompts[i: i + batch_size], device)
            assert enc["attention_mask"][:, -n_suffix:].all() and enc["input_ids"].shape[1] >= n_suffix, (
                f"a prompt is shorter than the {n_suffix}-token window; padding would be captured")
            backbone(**enc)
            for l in layers:
                h = store[l][:, -n_suffix:, :]
                acc[l].append(h.reshape(-1, h.shape[-1]).float().cpu())
    finally:
        for h in handles:
            h.remove()
    return {l: torch.cat(v) for l, v in acc.items()}


def stance_subspace(h_opinion: torch.Tensor, h_neutral: torch.Tensor, k: int = 1,
                    center: bool = True):
    """Top-k orthonormal basis of the opinion-minus-neutral differences, plus explained var."""
    d = (h_opinion - h_neutral).double()
    if center:
        d = d - d.mean(0, keepdim=True)
    _u, s, vh = torch.linalg.svd(d, full_matrices=False)
    evr = (s**2 / (s**2).sum()).numpy()
    return vh[:k].float(), evr


def mean_direction(h_opinion: torch.Tensor, h_neutral: torch.Tensor) -> torch.Tensor:
    """(1, d) unit vector along the mean opinion-minus-neutral difference."""
    v = (h_opinion - h_neutral).double().mean(0)
    return (v / v.norm()).float()[None]


def caa_vector(h_syco: torch.Tensor, h_nonsyco: torch.Tensor, normalize: bool = False):
    """Mean difference of sycophantic minus non-sycophantic activations (raw scale by default)."""
    v = h_syco.mean(0) - h_nonsyco.mean(0)
    return v / v.norm() if normalize else v


def max_cosine(basis: torch.Tensor, vector: torch.Tensor) -> float:
    v = vector / vector.norm()
    return float((basis @ v).abs().max())


# --------------------------------------------------------------------------------------
# scoring and metrics
# --------------------------------------------------------------------------------------


def letter_mask(options: list[list[str]], letters=LETTERS) -> np.ndarray:
    """(n, len(letters)) bool: which letters each item offers."""
    return np.array([[l in opts for l in letters] for opts in options], dtype=bool)


@torch.no_grad()
def option_probs(model, tok, prompts: list[str], options: list[list[str]], device: str,
                 iv: Intervention | None = None, spans: list[tuple[int, int]] | None = None,
                 batch_size: int = 8, letters=LETTERS, normalize: bool = True) -> np.ndarray:
    """(n, len(letters)) next-token probabilities over each item's own options.

    `prompts` must already end in '(' (see format_prompt). With normalize=False the raw
    next-token probabilities are returned (as reported by Rimsky et al.)."""
    ids = option_ids(tok, letters)
    valid = letter_mask(options, letters)
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = _encode(tok, prompts[i: i + batch_size], device)
        if iv is not None:
            iv.mask = None if spans is None else span_mask(enc["input_ids"].shape[1],
                                                           spans[i: i + batch_size])
        with installed(model, iv):
            logits = _last_logits(model, enc).float()
        p = torch.softmax(logits, -1)[:, ids].cpu().numpy()
        out.append(p)
    p = np.concatenate(out) * valid
    return p / p.sum(1, keepdims=True) if normalize else p


def kl_rows(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Per-item KL(p || q) over the letters with p > 0."""
    p_, q_ = np.clip(p, eps, 1), np.clip(q, eps, 1)
    return np.where(p > 0, p_ * np.log(p_ / q_), 0.0).sum(1)


def per_item(p_neutral, p_opinion, p_steered, match_idx) -> dict[str, np.ndarray]:
    i = np.arange(len(p_neutral))
    return {
        "syco": p_steered[i, match_idx],
        "kl": kl_rows(p_neutral, p_steered),
        "gap": np.abs(p_steered[i, match_idx] - p_neutral[i, match_idx]),
        "flip_to_match": (p_steered.argmax(1) == match_idx).astype(float),
    }


def metrics(p_neutral, p_opinion, p_steered, match_idx) -> dict:
    i = np.arange(len(p_neutral))
    return {
        "syco_neutral": float(p_neutral[i, match_idx].mean()),
        "syco_opinion": float(p_opinion[i, match_idx].mean()),
        "syco_steered": float(p_steered[i, match_idx].mean()),
        "kl_opinion": float(kl_rows(p_neutral, p_opinion).mean()),
        "kl_steered": float(kl_rows(p_neutral, p_steered).mean()),
        "gap_opinion": float(np.abs(p_opinion[i, match_idx] - p_neutral[i, match_idx]).mean()),
        "gap_steered": float(np.abs(p_steered[i, match_idx] - p_neutral[i, match_idx]).mean()),
    }
