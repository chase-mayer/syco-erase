#!/usr/bin/env bash
# Download the evaluation data. Raw data is git-ignored; run this after cloning.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW/sycophancy_eval"

echo "== Anthropic model-written evals (sycophancy) =="
for f in sycophancy_on_nlp_survey sycophancy_on_philpapers2020 sycophancy_on_political_typology_quiz; do
  curl -sSL -o "$RAW/$f.jsonl" \
    "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/$f.jsonl"
  echo "  $f.jsonl: $(wc -l < "$RAW/$f.jsonl") items"
done

echo "== SycophancyEval (Sharma et al., ICLR 2024) =="
for f in answer are_you_sure feedback; do
  curl -sSL -o "$RAW/sycophancy_eval/$f.jsonl" \
    "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/$f.jsonl"
  echo "  $f.jsonl: $(wc -l < "$RAW/sycophancy_eval/$f.jsonl") items"
done

echo "Done. Next: python -m src.data --all"
