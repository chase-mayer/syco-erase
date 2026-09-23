"""Residual-stream capture and intervention via PyTorch forward hooks.

Interventions implemented:
  * ``AddVector``   — CAA baseline: h <- h + alpha * v
  * ``ProjectOut``  — Belief-Erasure Steering: h <- h - P_S h  (S = stance subspace)
  * ``PatchFrom``   — oracle: replace h with activations cached from the neutral prompt

All of them operate on the output of ``model.model.layers[i]`` (Llama/Qwen/Gemma layout)
and can be restricted to a token-position slice, which is what BES needs: erase only over
the biography span.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


def layer_module(model, layer: int):
    """Return the decoder block whose output is the residual stream after `layer`."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer]
    raise ValueError(f"unsupported architecture: {type(model)}")


def _apply(out, fn):
    """Decoder blocks return a tensor or a tuple whose first element is the hidden state."""
    if isinstance(out, tuple):
        return (fn(out[0]),) + out[1:]
    return fn(out)


@dataclass
class Intervention:
    layers: list[int]
    positions: slice | None = None  # None => all positions

    def edit(self, h: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def _edit_slice(self, h: torch.Tensor) -> torch.Tensor:
        if self.positions is None:
            return self.edit(h)
        h = h.clone()
        h[:, self.positions, :] = self.edit(h[:, self.positions, :])
        return h


@dataclass
class AddVector(Intervention):
    """CAA (Rimsky et al., 2024): add a scaled behaviour vector."""

    vector: torch.Tensor = field(default_factory=lambda: torch.zeros(1))
    alpha: float = 1.0

    def edit(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.alpha * self.vector.to(h.device, h.dtype)


@dataclass
class ProjectOut(Intervention):
    """Belief-Erasure Steering: remove the stance subspace from the residual stream.

    ``basis`` is (k, d) with orthonormal rows. ``strength`` in [0, 1] interpolates between
    no erasure (0) and full erasure (1), which gives the Pareto sweep.
    """

    basis: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1))
    strength: float = 1.0

    def edit(self, h: torch.Tensor) -> torch.Tensor:
        b = self.basis.to(h.device, h.dtype)          # (k, d)
        coeff = torch.einsum("bld,kd->blk", h, b)      # projection coefficients
        return h - self.strength * torch.einsum("blk,kd->bld", coeff, b)


@dataclass
class PatchFrom(Intervention):
    """Oracle: overwrite activations with ones cached from the neutral prompt."""

    cache: dict[int, torch.Tensor] = field(default_factory=dict)
    _layer: int | None = None

    def edit(self, h: torch.Tensor) -> torch.Tensor:
        src = self.cache[self._layer].to(h.device, h.dtype)
        return src[:, -h.shape[1]:, :]


@contextmanager
def apply_intervention(model, iv: Intervention | None):
    """Context manager that installs `iv` on its layers and removes it on exit."""
    if iv is None:
        yield
        return
    handles = []
    for layer in iv.layers:
        def make_hook(layer_idx):
            def hook(_mod, _inp, out):
                if isinstance(iv, PatchFrom):
                    iv._layer = layer_idx
                return _apply(out, iv._edit_slice)
            return hook

        handles.append(layer_module(model, layer).register_forward_hook(make_hook(layer)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def capture(model, tokenizer, prompts: list[str], layers: list[int], device="cuda",
            last_token_only: bool = True) -> dict[int, torch.Tensor]:
    """Return {layer: (n, d)} residual activations (last token) or {layer: (n, L, d)}."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states  # tuple length n_layers+1, each (n, L, d)
    result = {}
    for layer in layers:
        h = hs[layer + 1]
        if last_token_only:
            idx = enc["attention_mask"].sum(1) - 1
            h = h[torch.arange(h.shape[0]), idx]
        result[layer] = h.detach().float().cpu()
    return result
