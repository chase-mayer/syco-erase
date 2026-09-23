"""Run the BES experiment end to end.

    python -m src.run_experiment --model Qwen/Qwen2.5-0.5B-Instruct --n 200 --layer 12

Steps
  1. sample items and split into fit / eval
  2. cache aligned suffix activations for opinion and neutral prompts  -> stance subspace
  3. cache answer-token activations for sycophantic / non-sycophantic completions -> CAA vector
  4. novelty check: max |cos| between the subspace and the CAA vector
  5. evaluate: neutral, opinion, BES at several strengths, CAA at several alphas
  6. write results/<tag>.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from . import bes

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def load_pairs(datasets: list[str], n: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    for name in datasets:
        path = PROCESSED / f"pairs_{name}.jsonl"
        rows += [json.loads(l) for l in path.open()]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--datasets", nargs="+", default=["nlp", "phil", "political"])
    ap.add_argument("--n", type=int, default=200, help="total items (fit + eval)")
    ap.add_argument("--fit-frac", type=float, default=0.5)
    ap.add_argument("--layer", type=int, default=None, help="default: 60% depth")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--window", type=int, default=16, help="aligned suffix tokens for fitting")
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--alphas", nargs="+", type=float, default=[-1.0, -2.0, -4.0])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, tok, device = bes.load(args.model, dtype=args.dtype)
    n_layers = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else int(0.6 * n_layers)
    opt = bes.option_ids(tok)
    print(f"model={args.model} layers={n_layers} steering layer={layer} device={device}")

    rows = load_pairs(args.datasets, args.n, args.seed)
    n_fit = int(len(rows) * args.fit_frac)
    fit, ev = rows[:n_fit], rows[n_fit:]
    print(f"items: {len(fit)} fit / {len(ev)} eval")

    # ---- 2. stance subspace ---------------------------------------------------------
    h_op = bes.suffix_activations(model, tok, [r["opinion_prompt"] for r in fit],
                                  [layer], args.window, device, args.batch_size)[layer]
    h_ne = bes.suffix_activations(model, tok, [r["neutral_prompt"] for r in fit],
                                  [layer], args.window, device, args.batch_size)[layer]
    basis, evr = bes.stance_subspace(h_op, h_ne, k=args.rank)
    print(f"stance subspace: rank={args.rank} variance explained={evr[:args.rank].sum():.3f}")

    # ---- 3. CAA behaviour vector ----------------------------------------------------
    syco = [r["opinion_prompt"] + " " + r["matching"] for r in fit]
    non = [r["opinion_prompt"] + " " + r["not_matching"] for r in fit]
    h_s = bes.suffix_activations(model, tok, syco, [layer], 1, device, args.batch_size)[layer]
    h_n = bes.suffix_activations(model, tok, non, [layer], 1, device, args.batch_size)[layer]
    v_caa = bes.caa_vector(h_s, h_n)

    # ---- 4. novelty check -----------------------------------------------------------
    cos = bes.max_cosine(basis, v_caa)
    print(f"NOVELTY CHECK  max|cos(stance subspace, CAA vector)| = {cos:.3f}")

    # ---- 5. evaluation --------------------------------------------------------------
    opinion_prompts = [r["opinion_prompt"] for r in ev]
    neutral_prompts = [r["neutral_prompt"] for r in ev]
    match_idx = np.array([0 if r["matching"].strip().startswith("(A") else 1 for r in ev])
    q_len = [len(tok.encode(p, add_special_tokens=False)) for p in neutral_prompts]

    p_neutral = bes.ab_probs(model, tok, neutral_prompts, opt, device,
                             batch_size=args.batch_size)
    p_opinion = bes.ab_probs(model, tok, opinion_prompts, opt, device,
                             batch_size=args.batch_size)

    results = {"config": vars(args) | {"layer": layer, "n_layers": n_layers},
               "novelty_cosine": cos,
               "variance_explained": float(evr[:args.rank].sum()),
               "conditions": {}}

    for s in args.strengths:
        iv = bes.ProjectOut(layers=[layer], basis=basis, strength=s)
        p = bes.ab_probs(model, tok, opinion_prompts, opt, device, iv=iv,
                         n_suffix=q_len, batch_size=args.batch_size)
        m = bes.metrics(p_neutral, p_opinion, p, match_idx)
        results["conditions"][f"bes_s{s}"] = m
        print(f"BES  s={s:<5} syco {m['syco_opinion']:.3f} -> {m['syco_steered']:.3f} "
              f"| KL {m['kl_opinion']:.4f} -> {m['kl_steered']:.4f} "
              f"| gap {m['gap_opinion']:.3f} -> {m['gap_steered']:.3f}")

    for a in args.alphas:
        iv = bes.AddVector(layers=[layer], vector=v_caa, alpha=a)
        p = bes.ab_probs(model, tok, opinion_prompts, opt, device, iv=iv,
                         n_suffix=q_len, batch_size=args.batch_size)
        m = bes.metrics(p_neutral, p_opinion, p, match_idx)
        results["conditions"][f"caa_a{a}"] = m
        print(f"CAA  a={a:<5} syco {m['syco_opinion']:.3f} -> {m['syco_steered']:.3f} "
              f"| KL {m['kl_opinion']:.4f} -> {m['kl_steered']:.4f} "
              f"| gap {m['gap_opinion']:.3f} -> {m['gap_steered']:.3f}")

    RESULTS.mkdir(exist_ok=True)
    tag = args.tag or f"{args.model.split('/')[-1]}_L{layer}_k{args.rank}_n{len(ev)}"
    out = RESULTS / f"{tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
