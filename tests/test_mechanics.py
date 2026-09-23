"""Mechanics tests that need no model download (run anywhere):

    python -m tests.test_mechanics

Checks the pieces that are easy to get silently wrong: hook installation/removal, the
suffix mask under left padding, the projection actually removing the subspace, and the
metric formulas.
"""

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src import bes


def tiny_model(d=32, layers=4, vocab=128):
    cfg = LlamaConfig(hidden_size=d, intermediate_size=2 * d, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=vocab)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_mask():
    m = bes.suffix_mask(3, 6, [2, 4, 10])
    assert m[0].tolist() == [0, 0, 0, 0, 1, 1]
    assert m[1].tolist() == [0, 0, 1, 1, 1, 1]
    assert m[2].all(), "n_suffix longer than sequence should clamp to full length"


def test_projection_removes_subspace():
    torch.manual_seed(0)
    d, k = 32, 2
    basis = torch.linalg.qr(torch.randn(d, k))[0].T          # (k, d) orthonormal rows
    iv = bes.ProjectOut(layers=[0], basis=basis, strength=1.0)
    h = torch.randn(2, 5, d)
    out = iv(h)
    residual = torch.einsum("bld,kd->blk", out, basis).abs().max()
    assert residual < 1e-5, residual
    half = bes.ProjectOut(layers=[0], basis=basis, strength=0.5)(h)
    assert torch.allclose(half, (h + out) / 2, atol=1e-5), "strength should interpolate"


def test_mask_restricts_edit():
    torch.manual_seed(0)
    d = 16
    basis = torch.linalg.qr(torch.randn(d, 1))[0].T
    iv = bes.ProjectOut(layers=[0], basis=basis, strength=1.0)
    iv.mask = bes.suffix_mask(1, 4, [2])
    h = torch.randn(1, 4, d)
    out = iv(h)
    assert torch.allclose(out[:, :2], h[:, :2]), "padded/left positions must be untouched"
    assert not torch.allclose(out[:, 2:], h[:, 2:]), "suffix positions must be edited"


def test_hook_changes_logits_and_is_removed():
    model = tiny_model()
    ids = torch.randint(0, 128, (2, 6))
    base = model(ids).logits.clone()
    v = torch.randn(model.config.hidden_size)
    with bes.installed(model, bes.AddVector(layers=[2], vector=v, alpha=5.0)):
        steered = model(ids).logits
    assert not torch.allclose(base, steered, atol=1e-4), "intervention had no effect"
    assert torch.allclose(base, model(ids).logits), "hook was not removed"


def test_subspace_recovers_planted_direction():
    torch.manual_seed(0)
    d, n = 64, 256
    direction = torch.nn.functional.normalize(torch.randn(d), dim=0)
    neutral = torch.randn(n, d)
    opinion = neutral + torch.randn(n, 1) * direction + 0.01 * torch.randn(n, d)
    basis, evr = bes.stance_subspace(opinion, neutral, k=1)
    assert bes.max_cosine(basis, direction) > 0.99, bes.max_cosine(basis, direction)
    assert evr[0] > 0.9


def test_metrics():
    p_neutral = np.array([[0.5, 0.5], [0.6, 0.4]])
    p_opinion = np.array([[0.9, 0.1], [0.2, 0.8]])
    match = np.array([0, 1])
    m = bes.metrics(p_neutral, p_opinion, p_neutral, match)
    assert abs(m["syco_opinion"] - 0.85) < 1e-9
    assert abs(m["syco_steered"] - 0.45) < 1e-9
    assert m["kl_steered"] == 0.0 and m["gap_steered"] == 0.0
    assert m["kl_opinion"] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
