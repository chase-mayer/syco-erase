# Data

The original datasets used in this project, unchanged from their public releases. The code reads
the files in `raw/` directly; `scripts/download_data.sh` skips any file that is already here.

| Folder / file | Source | Licence | Used in |
|---|---|---|---|
| `raw/sycophancy_on_nlp_survey.jsonl` (9,984 items) | Anthropic model-written evals, sycophancy (Perez et al., 2023), [github.com/anthropics/evals](https://github.com/anthropics/evals/tree/main/sycophancy) | CC BY 4.0 | 01–05, 07–09 |
| `raw/sycophancy_on_philpapers2020.jsonl` (9,867) | same | CC BY 4.0 | 01–05, 07–09 |
| `raw/sycophancy_on_political_typology_quiz.jsonl` (10,200) | same | CC BY 4.0 | 01–05, 07–09 |
| `raw/sycophancy_eval/answer.jsonl` | SycophancyEval (Sharma et al., 2024), [github.com/meg-tong/sycophancy-eval](https://github.com/meg-tong/sycophancy-eval) | MIT | factual control (04, 09) |
| `raw/sycophancy_eval/are_you_sure.jsonl` | same | MIT | pushback transfer (05) |
| `rimsky_caa/` | Rimsky et al. (2024) CAA release, [github.com/nrimsky/CAA](https://github.com/nrimsky/CAA) (commit `5dabbbd`) | MIT (`rimsky_caa/LICENSE`) | 06–09 |

`rimsky_caa/` keeps only the parts of that release this project uses, in their original paths:

- `datasets/generate/sycophancy/generate_dataset.json`: the 1,000 CAA fitting items
- `datasets/test/sycophancy/test_dataset_ab.json`: the 50 A/B test items
- `datasets/test/mmlu/mmlu.json`: the 570-item MMLU set
- `vectors_and_results.zip` (136 files, kept zipped), which unpacks to:
  - `normalized_vectors/sycophancy/`: the released Llama-2-7B-Chat sycophancy vectors, layers 0–31
  - `vectors/sycophancy/vec_layer_13_Llama-2-7b-chat-hf.pt`: the unnormalised layer-13 vector
  - `results/sycophancy/`: their published per-item Llama-2-7B-Chat results (A/B and MMLU)

Notebooks 06–09 clone the full CAA repository into `raw/caa_ref/` when run; `rimsky_caa/` is the
same content, kept here so the inputs are versioned with the results.

MMLU for the Qwen capability check (1,000 random test items, seed 0) is downloaded from Hugging
Face (`cais/mmlu`) on first use and cached as `raw/mmlu_sample.jsonl`.
`processed/` (built by `python -m src.data --dataset all`) holds the opinion/neutral prompt
pairs derived from `raw/`; it is regenerated, not stored.
