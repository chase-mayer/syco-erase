"""Uncertainty and comparison helpers.

Benchmark items repeat each question many times with different biographies, so items are
not independent. All intervals here resample whole questions (cluster bootstrap).
"""

from __future__ import annotations

from math import comb

import numpy as np


def _cluster_index(clusters):
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    return uniq, inv


def cluster_bootstrap(values, clusters, n_boot: int = 2000, seed: int = 0):
    """Mean of `values` with a 95% interval from resampling clusters. Returns (mean, lo, hi)."""
    values = np.asarray(values, dtype=float)
    uniq, inv = _cluster_index(clusters)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_cluster_bootstrap(a, b, clusters, n_boot: int = 2000, seed: int = 0):
    """Mean of (a - b) per item with a cluster-bootstrap 95% interval and two-sided p-value."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    uniq, inv = _cluster_index(clusters)
    sums = np.bincount(inv, weights=diff, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"diff": float(diff.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "p": float(min(p, 1.0)),
            "n_items": int(len(diff)), "n_clusters": int(len(uniq))}


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar test on paired binary outcomes."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p": 1.0}
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_a": only_a, "only_b": only_b, "p": float(p)}


def interp_at(x, y, targets):
    """Linear interpolation of y at each target x; NaN outside the observed x range."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    t = np.asarray(targets, float)
    out = np.interp(t, x, y)
    out[(t < x.min()) | (t > x.max())] = np.nan
    return out


def first_crossing(x, y, target):
    """Walk the curve in the order given; at the first segment where x crosses `target`,
    return (fraction along the segment index, interpolated y). (nan, nan) if never crossed."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(len(x) - 1):
        lo, hi = sorted((x[i], x[i + 1]))
        if lo <= target <= hi and x[i] != x[i + 1]:
            t = (target - x[i]) / (x[i + 1] - x[i])
            return i + t, float(y[i] + t * (y[i + 1] - y[i]))
    return float("nan"), float("nan")


def auc(scores, labels) -> float:
    """Rank-based ROC AUC (ties count half)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))
