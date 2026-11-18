"""Mechanics tests that need no download or GPU:

    python -m tests.test_mechanics

Covers the pieces that are easy to get silently wrong: hooks, span masks under left padding,
the projection and LEACE edits, multi-option scoring, question-grouped splits, metrics and
the statistics helpers.
"""

import random

import numpy as np
import torch

from src import bes, experiment, stats
from src.data import option_letters
from tests.tiny import tiny_model, tiny_tokenizer


def test_span_mask():
    m = bes.span_mask(6, [(2, 0), (4, 1), (10, 0)])
    assert m[0].tolist() == [0, 0, 0, 0, 1, 1]
    assert m[1].tolist() == [0, 0, 1, 1, 1, 0]
    assert m[2].all(), "a span longer than the sequence clamps to the full length"
    assert bes.suffix_mask(1, 4, [2])[0].tolist() == [0, 0, 1, 1]


def test_projection_removes_subspace():
    torch.manual_seed(0)
    d, k = 32, 2
    basis = torch.linalg.qr(torch.randn(d, k))[0].T
    h = torch.randn(2, 5, d)
    out = bes.ProjectOut(layers=[0], basis=basis, strength=1.0)(h)
    assert torch.einsum("bld,kd->blk", out, basis).abs().max() < 1e-5
    half = bes.ProjectOut(layers=[0], basis=basis, strength=0.5)(h)
    assert torch.allclose(half, (h + out) / 2, atol=1e-5), "strength should interpolate"


def test_mask_restricts_edit():
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(16, 1))[0].T
    iv = bes.ProjectOut(layers=[0], basis=basis, strength=1.0)
    iv.mask = bes.span_mask(4, [(2, 0)])
    h = torch.randn(1, 4, 16)
    out = iv(h)
    assert torch.allclose(out[:, :2], h[:, :2]), "positions outside the span must be untouched"
    assert not torch.allclose(out[:, 2:], h[:, 2:]), "span positions must be edited"


def test_hook_changes_logits_and_is_removed():
    model = tiny_model(vocab_size=128)
    ids = torch.randint(0, 128, (2, 6))
    base = model(ids).logits.clone()
    v = torch.randn(model.config.hidden_size)
    with bes.installed(model, bes.AddVector(layers=[2], vector=v, alpha=5.0)):
        steered = model(ids).logits
    assert not torch.allclose(base, steered, atol=1e-4), "intervention had no effect"
    assert torch.allclose(base, model(ids).logits), "hook was not removed"


def test_capture_matches_hook_layer():
    tok = tiny_tokenizer()
    model = tiny_model(len(tok))
    acts = bes.suffix_activations(model, tok, ["hello (", "hi there ("], [1], 3, "cpu", 2)[1]
    seen = {}
    handle = bes.layer_module(model, 1).register_forward_hook(
        lambda _m, _i, o: seen.setdefault("h", o[0] if isinstance(o, tuple) else o))
    model(**bes._encode(tok, ["hello (", "hi there ("], "cpu"))
    handle.remove()
    assert torch.allclose(acts, seen["h"][:, -3:].reshape(-1, acts.shape[-1]).float(), atol=1e-5)


def test_left_padding_does_not_change_scores():
    tok = tiny_tokenizer()
    model = tiny_model(len(tok))
    prompts = ["Q?\n (A) x\n (B) y\n\nAnswer: (", "A much longer question here?\n (A) x\n (B) y (C) z\n\nAnswer: ("]
    opts = [["A", "B"], ["A", "B", "C"]]
    batched = bes.option_probs(model, tok, prompts, opts, "cpu", batch_size=2)
    single = np.concatenate([bes.option_probs(model, tok, [p], [o], "cpu") for p, o in zip(prompts, opts)])
    assert np.allclose(batched, single, atol=1e-4)
    assert np.allclose(batched.sum(1), 1) and batched[0, 2:].sum() == 0, "only offered letters"


def test_subspace_recovers_planted_direction():
    torch.manual_seed(0)
    d, n = 64, 256
    direction = torch.nn.functional.normalize(torch.randn(d), dim=0)
    neutral = torch.randn(n, d)
    opinion = neutral + torch.randn(n, 1) * direction + 0.01 * torch.randn(n, d)
    basis, evr = bes.stance_subspace(opinion, neutral, k=1)
    assert bes.max_cosine(basis, direction) > 0.99
    assert evr[0] > 0.9
    shift = neutral + 3 * direction
    assert bes.max_cosine(bes.mean_direction(shift, neutral), direction) > 0.99


def test_leace_erases_linear_concept():
    from concept_erasure import LeaceEraser

    torch.manual_seed(0)
    z = torch.randint(0, 2, (500,)).float()
    x = torch.randn(500, 8) + 2 * z[:, None] * torch.tensor([1.0] + [0.0] * 7)
    er = LeaceEraser.fit(x, z[:, None])
    iv = bes.AffineErase(layers=[0], proj_left=er.proj_left, proj_right=er.proj_right, bias=er.bias)
    out = iv(x[None])[0]
    gap = out[z == 1].mean(0) - out[z == 0].mean(0)
    assert gap.abs().max() < 1e-3, "class means must coincide after LEACE"


def test_option_letters():
    assert option_letters("Q\n (A) no\n (B) yes\n\nAnswer:") == ["A", "B"]
    assert option_letters("(A)5\n(B)6\n(C)7") == ["A", "B", "C"]
    assert option_letters("I think (A) is right\n (A) x\n (B) y") == ["A", "B"]


def test_question_grouped_split_has_no_leakage():
    rows = [{"qid": f"{d}-{q}", "dataset": d} for d in ("a", "b") for q in range(10) for _ in range(5)]
    s = experiment.split_by_question(rows, seed=0, frac=(0.3, 0.2, 0.5))
    ids = {k: {r["qid"] for r in v} for k, v in s.items()}
    assert not (ids["fit"] & ids["val"]) and not (ids["fit"] & ids["test"]) and not (ids["val"] & ids["test"])
    assert sum(len(v) for v in s.values()) == len(rows)
    bal = experiment.sample_balanced(s["test"], 10, seed=0)
    assert len(bal) == 10 and {r["dataset"] for r in bal} == {"a", "b"}


def test_metrics():
    p_neutral = np.array([[0.5, 0.5, 0.0], [0.2, 0.2, 0.6]])
    p_opinion = np.array([[0.9, 0.1, 0.0], [0.1, 0.1, 0.8]])
    match = np.array([0, 2])
    m = bes.metrics(p_neutral, p_opinion, p_neutral, match)
    assert np.isclose(m["syco_neutral"], 0.55) and np.isclose(m["syco_opinion"], 0.85)
    assert np.isclose(m["kl_steered"], 0.0) and m["kl_opinion"] > 0
    assert np.isclose(m["gap_opinion"], 0.3)


def test_stats():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(20), 10)
    x = rng.normal(size=200)
    mean, lo, hi = stats.cluster_bootstrap(x, clusters)
    assert lo < mean < hi
    r = stats.paired_cluster_bootstrap(x + 1, x, clusters)
    assert np.isclose(r["diff"], 1) and r["p"] < 0.01
    assert stats.mcnemar_exact([1, 1, 1, 0], [1, 1, 1, 0])["p"] == 1.0
    assert stats.mcnemar_exact([1] * 12, [0] * 12)["p"] < 0.001
    assert np.isclose(stats.auc([0.1, 0.9], [0, 1]), 1.0)
    assert np.isnan(stats.interp_at([0, 1], [0, 1], [2])[0])
    pos, y = stats.first_crossing([0.7, 0.68, 0.6, 0.5], [0.3, 0.28, 0.2, 0.4], 0.64)
    assert np.isclose(pos, 1.5) and np.isclose(y, 0.24)
    assert np.isnan(stats.first_crossing([0.7, 0.68], [0.3, 0.28], 0.5)[0])


def test_window_longer_than_prompt_is_rejected():
    tok = tiny_tokenizer()
    model = tiny_model(len(tok))
    try:
        bes.suffix_activations(model, tok, ["short (", "a much longer prompt ("], [1], 12, "cpu", 2)
    except AssertionError:
        return
    raise AssertionError("padding inside the capture window must be rejected")


def test_format_prompt():
    tok = tiny_tokenizer()
    raw = bes.format_prompt(tok, "Q?\n\nAnswer:")
    assert raw.endswith("Answer: (")
    chat = bes.format_prompt(tok, "Q?\n\nAnswer:", chat=True, system="S")
    assert chat.endswith("<|assistant|>(") and "Answer:" not in chat and "<|system|>S" in chat


if __name__ == "__main__":
    random.seed(0)
    names = sorted(k for k in dir() if k.startswith("test_"))
    for name in names:
        globals()[name]()
        print("ok ", name)
    print(f"\n{len(names)} tests passed")
