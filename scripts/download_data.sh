#!/usr/bin/env bash
# Download the evaluation data (git-ignored). Files already present are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW/sycophancy_eval"

fetch() {
  if [ -s "$2" ]; then echo "  $(basename "$2"): present ($(wc -l < "$2") lines)"; return; fi
  curl -sSfL -o "$2" "$1"
  echo "  $(basename "$2"): $(wc -l < "$2") items"
}

echo "== Anthropic model-written evals (sycophancy; Perez et al., 2023) =="
for f in sycophancy_on_nlp_survey sycophancy_on_philpapers2020 sycophancy_on_political_typology_quiz; do
  fetch "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/$f.jsonl" "$RAW/$f.jsonl"
done

echo "== SycophancyEval (Sharma et al., 2024) =="
for f in answer are_you_sure feedback; do
  fetch "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/$f.jsonl" "$RAW/sycophancy_eval/$f.jsonl"
done
