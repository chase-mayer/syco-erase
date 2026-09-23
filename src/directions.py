"""Estimate (a) the CAA behaviour vector and (b) the BES stance subspace.

Behaviour vector (baseline, Rimsky et al. 2024):
    v = mean(h | sycophantic answer) - mean(h | non-sycophantic answer)

Stance subspace (this project):
    d_i = h(opinion_prompt_i) - h(neutral_prompt_i)   # same item, biography removed
    S   = top-k right singular vectors of the centred matrix D
Rank k is a hyper-parameter; k = 1 reduces to a difference-in-means direction. If the
``concept-erasure`` package is installed, ``leace_basis`` gives a whitened alternative that
guarantees linear non-recoverability of the stance label.

Experiment #1 in the proposal is ``cosine_to_behaviour``: if the stance subspace is parallel
to the CAA behaviour vector, BES collapses into prior work and we say so in the report.
"""

from __future__ import annotations

import numpy as np
import torch


def diff_in_means(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """(n,d),(m,d) -> (d,) unit vector pointing from neg to pos."""
    v = pos.mean(0) - neg.mean(0)
    return v / v.norm()


def stance_subspace(h_opinion: torch.Tensor, h_neutral: torch.Tensor, k: int = 1,
                    center: bool = True) -> tuple[torch.Tensor, np.ndarray]:
    """Top-k orthonormal basis of the opinion-minus-neutral difference space.

    Returns (basis (k,d), explained_variance_ratio).
    """
    d = (h_opinion - h_neutral).double()
    if center:
        d = d - d.mean(0, keepdim=True)
    _u, s, vh = torch.linalg.svd(d, full_matrices=False)
    evr = (s**2 / (s**2).sum()).numpy()
    return vh[:k].float(), evr


def leace_basis(h_opinion: torch.Tensor, h_neutral: torch.Tensor):
    """Closed-form linear concept erasure (Belrose et al., 2023), if available."""
    try:
        from concept_erasure import LeaceFitter
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install concept-erasure") from exc
    x = torch.cat([h_opinion, h_neutral]).double()
    z = torch.cat([torch.ones(len(h_opinion)), torch.zeros(len(h_neutral))]).double()[:, None]
    fitter = LeaceFitter(x.shape[1], 1, dtype=x.dtype)
    fitter.update(x, z)
    return fitter.eraser


def cosine_to_behaviour(basis: torch.Tensor, behaviour: torch.Tensor) -> float:
    """Largest |cos| between the behaviour vector and any direction in the subspace."""
    b = behaviour / behaviour.norm()
    return float((basis @ b).abs().max())


def subspace_report(basis: torch.Tensor, evr: np.ndarray, behaviour: torch.Tensor) -> str:
    return (
        f"rank={basis.shape[0]}  d={basis.shape[1]}\n"
        f"variance explained by top-{basis.shape[0]}: {evr[:basis.shape[0]].sum():.3f}\n"
        f"max |cos| with CAA behaviour vector: {cosine_to_behaviour(basis, behaviour):.3f}\n"
        "  (≈1.0 => BES reduces to behaviour steering; report this either way)"
    )
