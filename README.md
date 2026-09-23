# Belief-Erasure Steering (BES)

Removing the user's stated stance from the residual stream to mitigate sycophancy.

DATASCI 266 (NLP with Deep Learning), UC Berkeley MIDS — solo final project.
Proposal: [`docs/proposal.md`](docs/proposal.md).

## Idea in one paragraph

Language models shift their answers toward opinions users state about themselves. Existing
inference-time fixes steer the model's *output behaviour* (Contrastive Activation Addition
adds an "anti-sycophancy" vector at every token), which can overshoot into contrarianism and
has no principled stopping point. BES instead removes the model's internal representation of
**what the user said they believe**: it estimates a low-rank stance subspace from prompt
pairs that differ only in the user's biography, then projects that subspace out of the
residual stream over the biography tokens. The target is invariance — behave as if no
opinion had been stated — so the same item with the biography deleted provides an oracle
reference distribution, and success is a distance to that distribution rather than a judge's
rating.

## Data

| Source | Items | Use |
|---|---|---|
| `sycophancy_on_nlp_survey.jsonl` | 9,984 | main A/B evaluation + pair construction |
| `sycophancy_on_philpapers2020.jsonl` | 9,867 | main A/B evaluation + pair construction |
| `sycophancy_on_political_typology_quiz.jsonl` | 10,200 | main A/B evaluation + pair construction |
| SycophancyEval `answer` / `are_you_sure` / `feedback` | 7,267 / 4,887 / 8,500 | generalisation to other sycophancy types |
| MMLU subset | 570 (10 × 57) | capability retention |

The first three are Anthropic's model-written evaluations (Perez et al., 2023) — the same
data the CAA paper steered on. Raw data is git-ignored; fetch it with:

```bash
bash scripts/download_data.sh
python -m src.data --all          # writes data/processed/pairs_*.jsonl
```

Each processed row carries `opinion_prompt` (original item), `neutral_prompt` (biography
removed), `bio` (the erasure span), and the user-matching / non-matching answer letters.
`split_ok=false` flags items where the question-stem heuristic fell back to a sentence
split — inspect those with `python -m src.data --dataset political --review 25`.

## Method

1. Cache residual activations for the opinion and neutral prompts at layers L (start at the
   middle-to-late band; prior work localises the effect around layers 19–23 in 7–8B models).
2. Fit the stance subspace: SVD of the (opinion − neutral) difference matrix, or LEACE for a
   whitened projector (`src/directions.py`).
3. **Novelty check first:** cosine similarity between the stance subspace and the CAA
   behaviour vector. If ≈1, BES reduces to prior work — report it.
4. Install `ProjectOut` over the biography token span and sweep `strength` 0 → 1
   (`src/hooks.py`).
5. Score A/B options and compute sycophancy rate, KL to the neutral distribution, and the
   invariance gap (`src/evaluate.py`).

## Baselines

| Condition | Why |
|---|---|
| No intervention | floor |
| Prompt ("ignore my stated opinion") | AxBench showed prompting beats many steering methods |
| CAA behaviour vector, α swept | the method being improved on |
| Behaviour-direction projection ablation | separates "which direction" from "add vs. project" |
| Activation patching from the neutral prompt | **oracle upper bound**; needs the paired prompt BES does not |

Every method is swept and compared as a Pareto frontier (sycophancy reduction vs. MMLU and
perplexity), never at a single steering strength, and reported with per-item variance across
≥3 seeds.

## Models

`google/gemma-2-2b-it` (fits a 16 GB GPU in bf16) and `Qwen/Qwen2.5-7B-Instruct`
(A100 on Colab, or 4-bit locally). Intervention is inference-only — no training.

## Layout

```
docs/proposal.md      proposal + why this is a new algorithm
scripts/              data download
src/data.py           opinion/neutral pair construction
src/hooks.py          activation capture, AddVector / ProjectOut / PatchFrom
src/directions.py     behaviour vector, stance subspace, LEACE, novelty check
src/evaluate.py       A/B scoring, sycophancy rate, KL, invariance gap
```

## Key references

- Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*, ACL 2024.
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, NeurIPS 2024.
- Belrose et al., *LEACE: Perfect Linear Concept Erasure in Closed Form*, NeurIPS 2023.
- Sharma et al., *Towards Understanding Sycophancy in Language Models*, ICLR 2024.
- Tan et al., *Analysing the Generalisation and Reliability of Steering Vectors*, NeurIPS 2024.
- Perez et al., *Discovering Language Model Behaviors with Model-Written Evaluations*, ACL Findings 2023.
- Wang et al., *When Truth Is Overridden*, arXiv:2508.02087 (motivation; oracle patching result).
