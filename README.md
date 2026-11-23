# Belief-Erasure Steering (BES)

Removing the user's stated stance from the residual stream to mitigate sycophancy.
DATASCI 266 (NLP with Deep Learning), UC Berkeley MIDS — solo final project, Chase Mayer.

## Idea

Language models shift their answers toward opinions users state about themselves.
Contrastive Activation Addition (CAA; Rimsky et al., 2024) counters this by *adding* an
anti-sycophancy behaviour vector to the residual stream. BES instead *removes* the model's
representation of what the user said they believe. Each benchmark item gives a pair of prompts
that share the same question: with the user's biography (opinion prompt) and without it
(neutral prompt). BES fits a direction to the position-aligned opinion-minus-neutral activation
differences (top-k right singular vectors U of the centred differences) and projects them out
at inference over the question tokens:

    h ← h − s · UᵀU h,   s ∈ [0, 1]

Because the neutral prompt exists for every item, success has a reference point: the
intervened model should answer as it does when no opinion is stated. The main metric is
KL(no-opinion ‖ intervened) alongside the probability of the user's preferred answer.

## Results in brief

### Qwen2.5-1.5B-Instruct, held-out questions, 3 seeds

| | Sycophancy | KL to no-opinion | MMLU | Acc. when user is wrong |
|---|---|---|---|---|
| No intervention | 0.709 | 0.364 | 0.595 | 0.128 |
| "Ignore my opinion" prompt | 0.742 | 0.527 | 0.588 | 0.170 |
| BES (selected: layers 11–13, rank 2, s = 0.75) | 0.510 | 0.150 | 0.526 | 0.406 |
| CAA matched to BES (layer 18, α = −13.0) | 0.519 | 0.288 | 0.370 | 0.330 |
| CAA at its best KL (α = −4) | 0.553 | 0.160 | 0.572 | 0.146 |
| Oracle (no-opinion prompt) | 0.470 | 0 | — | 0.628 |

- At matched sycophancy BES has lower KL than CAA in all three seeds (significant in two) and
  loses 6.9 MMLU points against CAA's 22.5. At mild reductions CAA is as good on KL and cheaper.
- The erased direction does **not** encode which answer the user favours (AUC ≈ 0.5) and works
  at layer 12, before that information is linearly readable; it removes the biography's imprint
  on the question tokens, not a stance axis.
- BES transfers to a factual control (user-wrong accuracy 0.13 → 0.41) but not to "are you
  sure?" pushback, where every method increases answer flipping.
- The CAA implementation reproduces Rimsky et al.'s published Llama-2-7B-Chat numbers within
  0.0013 (notebook 06, official Meta weights).
- **The result does not transfer to the public benchmark.** Run under Rimsky et al.'s own
  protocol on Llama-2-7B-Chat (their fitting items, test items, released vectors and metric;
  notebook 07), BES leaves sycophancy unchanged at every layer tried (0.695 → 0.711 on their
  50 items; 0.782 → 0.725 on 400 held-out PhilPapers items), while their layer-13 vector at
  α = −1 reaches 0.539 (published 0.538) and 0.525, and layer 12 reaches 0.466 (published
  0.465).
- **Why, and the fix (notebooks 08–09).** Activation patching shows the biography's influence
  runs through the question tokens and reaches the answer position between layers 12 and 16, as
  BES assumes, and that erasing it there would remove it entirely (0.43 vs no-opinion 0.43). The
  rank was the problem: on Llama-2 the imprint is spread over many directions (rank 2 / 4 / 8 / 16
  at layer 12: 0.726 / 0.674 / 0.572 / 0.515 on validation). With clean selection on half the
  PhilPapers questions and testing on the other half, rank-16 SVD reaches official CAA (0.524 vs
  0.504, KL 0.50 vs 0.51, MMLU −3.9 vs −3.2 points) and four directions learned on the KL
  objective (model frozen, 450 steps) reach the no-opinion level (0.441; oracle 0.427) with CAA's
  KL at matched sycophancy (0.395 vs 0.440, n.s.) and no measurable MMLU cost (−0.7 points vs
  −6 to −8 for matched CAA). Notebook 08's five effect-selected directions did not replicate on the
  clean split (0.604). On Qwen, none of the alternatives beats the original setting.
- **Six current models, pre-registered protocol (notebook 10b).** On Qwen2.5-1.5B, Qwen3.5-2B,
  Qwen3.5-9B, Llama-3.2-3B, Llama-3.1-8B and Llama-2-7B, with every setting fixed before any
  model was run and CAA given at least BES's search, the learned eraser (k = 4) has lower KL than
  matched CAA on all three Qwen models (e.g. 0.031 vs 0.071 on Qwen3.5-2B, p < 0.001), level KL
  on Llama-3.2 and Llama-3.1, changes the answers to no-opinion prompts less on all six, and keeps
  12–25 more MMLU points on four. The training-free SVD is only on a par with CAA. The protocol's
  format check found that chat-template prompts ending in `(` put almost no probability on the
  answer letters, which invalidated notebook 10's chat-format runs.

## How the project evolved

Each notebook answers a question the previous one raised. Throughout, settings are chosen on
validation questions only, CAA always gets at least the search BES gets, and test questions are
scored once.

1. **01 (data).** The 30,051 items turned out to cover only 158 questions, so every split is
   grouped by question; many items have more than two options, so every metric uses each item's
   own options. The BES direction was found not to encode the user's answer.
2. **02 (BES ablations) → 03 (CAA steel man).** BES's layer, rank, estimator, span, window, band
   and strength were chosen on validation. A BES comparison is only meaningful against CAA at full
   strength, so 03 gave CAA its own layer, span and multiplier search on 1,000 fitting items and
   matched it to BES's sycophancy.
3. **04–05 (Qwen test results, pushback).** BES beat matched CAA on KL and MMLU on held-out
   questions; neither method helped with "are you sure?" pushback.
4. **06 (CAA against published results).** To rule out a weak or buggy baseline, the CAA code
   was run on Rimsky et al.'s release and reproduced their numbers within 0.0013.
5. **07 (BES on Llama-2 under the published protocol).** BES did nothing there while published
   CAA worked: the Qwen result did not transfer.
6. **08 (why).** Activation patching showed the target was right (erasing the question tokens at
   layers 10–16 would remove the biography's influence entirely), so the problem was the
   directions. Effect-selected directions helped, but were chosen on one draw of items from the
   fitting questions.
7. **09 (clean test, and the fix).** With selection and testing on disjoint PhilPapers
   questions, the real problem was rank (rank 2 removes almost nothing on Llama-2, rank 16
   reaches official CAA), and a learned eraser (same projection, directions fitted to the KL
   objective with the model frozen) reached the no-opinion level with CAA's KL and no MMLU cost.
8. **10 (current models).** Two gaps remained: the learned eraser and a full rank sweep had not
   been run on Qwen, and all results used 2024-era models. Notebook 10 ran both methods on
   current Qwen and Llama models with ranks to 256.
9. **10b (pre-registered protocol).** Notebook 10 raised concerns: a suspiciously low MMLU on
   Qwen3.5-2B, a learned eraser that damaged MMLU on Llama-3.1-8B, BES fitted on fewer items than
   CAA (300 vs 1,000), and selections at the edge of their grids. 10b fixed every choice in
   advance and widened the search for both methods: a prompt-format check, BES on CAA's 1,000
   items, finer layers, window, span, estimator and ranks to the cap, a retain term for the learned
   eraser, and more layers and multipliers for CAA. The format check showed notebook 10's
   chat-format prompts did not measure the models' answers, so 10b is the result for current
   models; it adds Qwen3.5-9B and Llama-2 under the unchanged protocol.

The written report was submitted separately through the course and is not part of this repo;
references to "the report" below point to it.

## Notebooks

The notebooks in `notebooks/` are the executed runs, with their outputs. Each code cell opens with a one-line
comment saying what it does; each section has a one-sentence summary. This section explains
what each notebook is for, how to read its output, what it found, and why the next one exists.

| # | Notebook | Model | Data | Writes to `results/` |
|---|---|---|---|---|
| 01 | `01_data_and_diagnostics` | Qwen | fit + validation | `data_overview`, `baseline_by_dataset`, `layer_diagnostics`, `novelty_check`, `fig_diagnostics` |
| 02 | `02_bes_ablations` | Qwen | validation | `ablation_*`, `fig_ablations`, `selected_config.json["bes"]` |
| 03 | `03_caa_steelman` | Qwen | validation | `caa_grid`, `caa_fit_size`, `caa_matched`, `prompt_baseline_val`, `fig_caa_steelman_val`, `selected_config.json["caa"]` |
| 04 | `04_main_results` | Qwen | **test**, 3 seeds | `test_*`, `table1_frontier`, `table2_control_capability`, `fig1_frontier`, `fig2_control_capability` |
| 05 | `05_transfer_pushback` | Qwen | pushback items | `test_pushback`, `fig3_pushback` |
| 06 | `06_caa_reproduction_llama2` | Llama-2-7B-Chat | Rimsky et al. release | `caa_reproduction_llama2` |
| 07 | `07_bes_llama2_vs_published_caa` | Llama-2-7B-Chat | Rimsky et al. release + PhilPapers | `llama_*` (no suffix), `fig_llama_frontier` |
| 08 | `08_bes_llama2_improvements` | Llama-2, then Qwen | as 07; Qwen seed 0 | `llama_imp_*`, `qwen_imp_*`, `fig_llama_diagnostics`, `fig_llama_improved`, `fig_qwen_improved` |
| 09 | `09_bes_llama2_closing_the_gap` | Llama-2, then Qwen | PhilPapers split by question; Qwen seed 0 | `llama_gap_*`, `qwen_gap_*`, `fig_llama_gap` |
| 10 | `10_bes_vs_caa_current_models` | Qwen2.5, Qwen3.5, Llama-3.2, Llama-3.1 | seed-0 splits, **test** | `m10_<model>_*`, `m10_summary`, `fig_m10_*` |
| 10b | `10b_bes_vs_caa_current_models_v2` | as 10, then Qwen3.5-9B, Llama-2 | seed-0 splits, **test** | `m10b_<model>_*`, `m10b_summary`, `m10b_v1_vs_v2`, `fig_m10b_*` |

**Terms used below.** *Sycophancy* (or `syco`, P) is the mean probability the model gives to the
user's preferred option; lower is better, down to the *oracle* level, which is the same quantity
on the neutral prompt. *KL* is KL(no-opinion ‖ intervened) over the item's options: 0 means the
method restored the no-opinion answers exactly. *s* is BES strength (fraction of each direction
removed), *α* is the CAA multiplier (negative pushes away from sycophancy), *rank* is the number
of directions removed, *span* is which tokens are edited. The *matched point* is the CAA
multiplier that gives the same sycophancy as the selected BES setting. Intervals are 95%
question-clustered bootstrap; MMLU differences use exact McNemar tests.

### 01 — Data and diagnostics

**Purpose.** Establish what the benchmark contains and what the BES direction encodes, before
any steering result is trusted.

**How to read it.** §1 counts items, distinct questions and option positions per subset. §2
builds the question-grouped splits. §3 is the size of the effect to be removed: P(user's
answer) with and without the biography. §4 asks whether the top BES direction carries the
user's stance: an AUC near 0.5 means it does not; the probe columns say at which layer the
user's option becomes linearly readable, and whether removing the direction changes that. §5
compares the BES directions with the CAA vector; |cos| near 1 would make BES a re-derivation of
CAA.

**Findings.** 30,051 items but only 158 distinct questions; 59% of PhilPapers items have more
than two options and 27% put the user's answer at (C)–(G). On validation the biography raises
P(user's answer) from 0.488 to 0.716 (+0.229, CI 0.160–0.319): +0.379 PhilPapers, +0.235 NLP,
+0.071 political (CI includes zero). The direction does not separate users favouring (A) from
(B) at any layer (AUC 0.47–0.55); the probe is at chance through layer 12 and reaches 0.84 by
layer 18; |cos| with CAA ≤ 0.23.

**What it led to.** The few-questions finding changed the whole pipeline to question-grouped
splits and multi-option scoring. The AUC result is why the report describes BES as removing
the biography's imprint on the question rather than a stance axis.

### 02 — BES ablations

**Purpose.** Choose every BES setting on validation questions only, by the objective BES is
built for: KL to the no-opinion answers.

**How to read it.** Sections A–G vary one setting at a time, each starting from the winner of
the previous one: layer, rank, estimator, span, fitting window, band of neighbouring layers,
strength. Each printed line gives sycophancy and KL for that setting; the dashed line in the
figure is no intervention (KL 0.332). The final cell writes the chosen setting to
`selected_config.json`.

**Findings.** Layer 12 (KL 0.187), rank 2 (0.091), centred SVD over mean difference (0.312)
and LEACE (0.271), question span (0.091; last token 0.297, biography 0.244), window 16, band ±1
(0.075), strength 0.75 (KL 0.068, sycophancy 0.521).

**What it led to.** This configuration is used unchanged in 03–05.

### 03 — CAA steel man and the prompting baseline

**Purpose.** Make sure the comparison method is at full strength.
Earlier runs disadvantaged CAA (150 fitting items, unit-normalised, wrong readout position, BES's
layer); here it follows Rimsky et al.'s recipe and gets its own search.

**How to read it.** §A is the full grid of layer × multiplier × span on validation, and the
selection of the setting with the lowest KL at BES's sycophancy level, plus the multiplier that
reaches it. §B checks whether fitting on 999 rather than 300 items matters. §C is the
"ignore my opinion" instruction.

**Findings.** Only 2 of 21 layer–span settings reach BES's sycophancy level, both at layer 18;
the selected one (all tokens) has its best KL at α = −4 (sycophancy 0.562, KL 0.157) and matches
BES at α = −13.0 (KL 0.240). Fit size changes little. The prompt raises sycophancy.

**What it led to.** The matched multiplier is written to `selected_config.json` for 04–05.

### 04 — Main results on held-out test questions

**Purpose.** The headline comparison: the validation-selected settings applied once to test
questions, for three seeds that each redraw the splits and refit every direction.

**How to read it.** §1–2: the frontier table (sycophancy and KL for every method and strength)
and paired differences at the matched point. Seed 0's test questions are fully disjoint from
selection; seeds 1–2 reuse seed-0 settings, so they show variability rather than independent
confirmation. §3 breaks seed 0 down by subset and by item. §4 is the factual control: a good
method should raise accuracy when the user endorses a wrong answer without lowering it when
the user is right. §5 is MMLU with each method applied.

**Findings.** Both methods cut sycophancy by about 0.20. At the matched point BES has lower KL
in all three seeds (−0.066, p = 0.35 on seed 0; −0.121 and −0.225, p ≤ 0.002 on seeds 1–2), and
loses 6.9 MMLU points to CAA's 22.5; at α = −4 CAA loses only 2.3. BES lifts accuracy under a
wrong user opinion from 0.128 to 0.406; CAA does not. BES raises KL on political items (0.146 →
0.393), driven by 165 items where the biography had *lowered* agreement.

**What it led to.** The report's main Qwen result (Tables 1–2, Figures 1–2).

### 05 — Transfer to conversational pushback

**Purpose.** Test whether directions fitted on stated opinions also help when the user objects
to the model's answer ("are you sure?").

**How to read it.** `flip_from_correct` is the harm (a right answer abandoned);
`flip_to_correct` is a legitimate correction. A method that raises both is adding instability
rather than reducing deference.

**Findings.** Nothing transfers: abandoning a correct answer rises from 0.265 with no
intervention to 0.440 (BES) and 0.443 (matched CAA), and corrections rise too.

**What it led to.** Reported as a limitation: pushback is a different mechanism.

### 06 — CAA implementation checked against Rimsky et al. (2024)

**Purpose.** Benchmark the comparison method against published results.
Their released Llama-2-7B-Chat vectors, data and per-item results are run through this repo's
hook and scoring code.

**How to read it.** Check 1 puts their vector through our code; check 2 extracts our own vector
from their data. `abs_diff` against their published probabilities should be about 0.01 or less,
and the cosine between vectors near 1.

**Findings.** Agreement within 0.0013 at every multiplier; cosine 1.0000.

**What it led to.** The Qwen CAA baseline uses a verified implementation. But 06 checks CAA's
code, not BES against a published number, which is what 07 adds.

### 07 — BES on Llama-2-7B-Chat against published CAA

**Purpose.** Put BES on the exact setup Rimsky et al. used (model, chat prompt, their 1,000
fitting items, their 50 test items, their metric, their released vectors for every layer, their
570-item MMLU set) so it can be read directly against published numbers.

**How to read it.** In the test table, `published_raw` / `published_norm` are their numbers for
the same items; for `none` and the CAA rows they should match ours to about 0.002, which confirms
the protocol. Their test questions all appear in their fitting set, so the 400 PhilPapers items
(questions neither method was fitted on) are the clean test. Note: BES never got below
no-intervention sycophancy, so the "matched" CAA rows in `llama_bes_vs_caa.csv` and
`llama_comparisons.csv` fell back to the strongest grid multiplier and are not a matched
comparison; they are not used in the report.

**Findings.** Protocol reproduced (within 0.002; MMLU within 0.005). BES as specified does not
reduce sycophancy on Llama-2: no validation setting moved it (0.746–0.762 vs 0.749), and on test
it goes from 0.695 to 0.711 on their items and 0.782 to 0.725 on PhilPapers. Their layer-13
vector at α = −1 gives 0.539 (published 0.538); layer 12 gives 0.466 (published 0.465).

**What it led to.** Notebook 08: why does a method that works on Qwen do nothing here?

### 08 — Why BES fails on Llama-2, and general improvements

**Purpose.** Part A diagnoses the Llama-2 failure and tries repairs. Part B tests whether any
of the improvements suggested by the Qwen results help on Qwen itself.

**How to read it.** §1 is the key diagnostic. Activation patching copies the no-opinion
prompt's activations into the opinion prompt at one layer, over the question tokens or the last
token only; that is the ceiling for any eraser at that layer and span. If patching the question
tokens restores the no-opinion answer, BES targets the right place. §2–3 is a full grid of
layer × rank × span under a stricter selection rule (a setting must lower sycophancy by 0.03,
then lowest KL). §4–6 test directions chosen by effect, directions aimed at the user's option,
and projecting out CAA's own vector. §7 tests the best of each. In Part B every row is compared
with the original setting (`p_*_vs_fixed`), not just with no intervention.

**Findings.** Patching the question tokens brings sycophancy from 0.763 to 0.632 / 0.562 /
0.518 / 0.492 at layers 8 / 10 / 12 / 14 (no-opinion 0.478); patching the last token does
nothing until layer 14. So the target is right. Rank-1 erasure does nothing at any layer, and
only 7 of 108 grid settings qualify. At layer 12 the top-variance direction has no effect, while
five lower-variance ones chosen by effect give 0.545–0.564 on test. Label-aimed erasure and
erasing the CAA vector do not work. Part B: per-item strength, effect selection, label-aimed
erasure do not beat the original on Qwen. Part B's chat-template check is not valid: notebook
10b showed that in Qwen2.5's chat format the prompt puts almost no probability on the answer
letters, so those scores do not measure the model's answer. (§8's table is empty in this run because the notebook-02 CSVs were not in
Colab's `results/`; `results/qwen_imp_selection_rule.csv` was regenerated from the committed
notebook-02 tables and shows every Qwen choice unchanged.)

**What it led to.** The effect-selected directions were chosen on validation items from the
fitting questions, on one seed, and still sat above the patching ceiling (0.571 vs 0.518 on
validation). Notebook 09 tests them cleanly and tries to close the gap.

### 09 — Closing the gap on Llama-2, with a generality check on Qwen

**Purpose.** Select every setting on questions that neither fitting nor testing uses, and find
out whether BES on Llama-2 can reach the patching ceiling and a Qwen-like advantage over CAA.

**How to read it.** The 109 PhilPapers questions are split in half: settings are chosen on
`phil_val` and tested on `phil_test` (no shared questions). §1 gives the reference points:
CAA, notebook 08's directions, and the patching ceiling. §2 compares estimators needing no
search across ranks. §3 is greedy effect selection with a stability check (Jaccard overlap of
helpful directions across validation draws). §4 combines layers. §5 trains a *learned eraser*:
four orthonormal directions fitted by gradient descent with the model frozen to minimise the
same KL objective, then applied by the same projection. §6 tests everything, with CAA matched
on `phil_val`, plus MMLU and the factual control. Part B repeats the estimators on Qwen.

**Findings.** The patching ceiling at layer 12 is 0.430 (oracle 0.427). Notebook 08's
directions give 0.604 on the clean split, so that result was partly draw-dependent. The real
problem was rank: sycophancy falls steadily with rank (0.726 / 0.674 / 0.572 / 0.515 at ranks
2 / 4 / 8 / 16 on validation). On `phil_test`, rank-16 SVD gives 0.524 (KL 0.50) against
official CAA's 0.504 (0.51), and the learned eraser gives 0.441 (KL 0.395), with KL equal to
layer-13 CAA at matched sycophancy (0.440, p = 0.55), half of layer-12 CAA's (0.890), and an MMLU
cost of 0.7 points against 6–8 for matched CAA. The factual control is a trade-off, not a win:
the learned eraser keeps 0.848 accuracy when the user is right and lifts the user-wrong case
from 0.046 to 0.248, while matched CAA reaches 0.44–0.47 there but drops to 0.62–0.67 when the
user is right. On Qwen nothing beats the original: greedy selection picks the original two
directions, and higher rank hurts.

**What it means.** The projection and the objective transfer across models; the rank-2 SVD
does not. Rank must be part of the validation search, and the learned eraser is the version of
BES that works on Llama-2; it was not run on Qwen here, and Qwen's SVD sweep stopped at rank 8. It is one run on one seed.

**What it led to.** Notebook 10: the full rank sweep and the learned eraser on Qwen, and both methods on current models.

### 10 — BES vs CAA on current Llama and Qwen models, with the full rank sweep

**Purpose.** Two gaps: on Qwen, notebook 09 swept SVD ranks only up to 8 and never trained the
learned eraser, and every result so far uses 2024-era models. This notebook runs one protocol,
unchanged, on Qwen2.5-1.5B-Instruct (the original model), Qwen3.5-2B (current Qwen at that
size), Llama-3.2-3B-Instruct (the newest Llama text model under 8B) and Llama-3.1-8B-Instruct
(the largest current Llama that fits one GPU, and Llama-2-7B's size). Qwen3.5-9B,
Llama-2-7B-Chat and Muse Glimmer 30B are listed as optional and were not run here; the first two
were run in notebook 10b. The notebook opens with a note that only its Qwen2.5 results are valid
(see Findings).

**Protocol (identical for every model).** Seed-0 question-grouped splits: 300 fit items for
BES, 1,000 for CAA (from the same fit questions), 450 validation, 900 test. BES: every layer in
the notebook 03 grid at ranks 4/16/64, then ranks 1–256 × strengths 0.5/0.75/1 at the chosen
layer (plus a ±1-layer band for the best three), then learned erasers with k = 4 and 16. CAA:
every layer × span × multiplier (−0.5 to −32). Settings are chosen on validation by lowest
sycophancy + KL among those lowering sycophancy by at least 0.03; CAA is matched on validation
to each headline BES setting's sycophancy, as in notebook 03. The test questions are scored once
for every method, with question-cluster bootstrap intervals and paired tests against no
intervention and against matched CAA, plus MMLU (1,000 items, McNemar) and the factual control.
Prompts use each model's chat template (thinking off for Qwen3.5), except Qwen2.5, which keeps
the raw format and fp32 of notebooks 02–05.

**How to read it.** `m10_summary` has one row per model: test sycophancy and KL for the
selected SVD setting and each learned eraser, the same for CAA matched to it, the KL p-value
against matched CAA, and MMLU drops. `m10_<model>_test` has every condition, including the
whole rank curve (`kind == "rank curve"`); `fig_m10_<model>` plots it with the matched CAA curve,
and `fig_m10_rank_curves` puts the models side by side.

**Findings.** Only the Qwen2.5-1.5B results are valid measurements. Notebook 10b's format
check showed that in the chat format used here for Qwen3.5-2B, Llama-3.2-3B and Llama-3.1-8B the
models put almost no next-token probability on the answer letters (under 1%, against 81–100% in
the raw `Answer: (` format). The prompt ends in a bare `(` token, which these tokenizers merge with
the letter that follows, so the scored letter tokens are ones the model would not produce there,
and renormalising over them compares leftover probabilities rather than the model's answers. The
same check put 98% of the mass on the letters for Qwen2.5 in its raw format, so its row stands:
the learned eraser (k = 4) has lower test KL than matched CAA (0.094 vs 0.331, p < 0.001) and
keeps more MMLU (0.574 vs 0.470; no intervention 0.595). For the other three models, use
notebook 10b, which replaces these runs.

**What it led to.** Notebook 10b. Before its format check, the run already raised concerns:
Qwen3.5-2B's baseline MMLU was implausibly low (0.357), the learned eraser damaged MMLU on
Llama-3.1-8B, BES was fitted on 300 items against CAA's 1,000, and some selections sat at the
edge of their grids. 10b fixes every choice in advance and widens the search for both methods.

### 10b — Protocol v2: the notebook 10 comparison with a wider, pre-specified search

**Purpose.** A second protocol, every setting fixed in the notebook's first settings cell before
any model was run; choices are made on validation and the test questions are scored once per
model. It was planned as a robustness check on notebook 10; because its format check found that
notebook 10's chat-format runs did not measure the models' answers (see notebook 10), it is the
result for the current models, and notebook 10 stands only for Qwen2.5.

**What changes (both methods get a wider search).** A prompt-format check before anything is
fitted (answer-letter mass, and whether batch padding changes the answers, on fitting prompts
and MMLU items 1000–1199, which are disjoint from the 1,000 evaluation items; a model that fails
is scored one prompt at a time, or switched format by a fixed rule). BES fits on the same 1,000
items as CAA (300 in v1); the layer grid adds the ±1 and ±2 neighbours of the best layer; the
question window is chosen from 8/16/32 tokens and the edited span from question or all tokens;
the estimator is chosen from centred SVD, mean difference and LEACE (notebook 02's ablation);
ranks run to 512 plus the cap (a quarter of the hidden size); the learned eraser's layer is chosen by a short training run at the three best
screened layers, and its loss adds a fixed-weight retain term (KL on the no-opinion prompts with
the eraser applied to every position, from fitting items only). CAA gets every layer BES saw plus
the ±1 and ±2 neighbours of its own best layer, and multipliers to −64 (10 values). A new test
metric, KL on the no-opinion prompts with each method applied, measures collateral change where
there is no opinion to remove, and notebook 01's novelty check (largest cosine between the BES
directions and the CAA vector at the same layer) is repeated on every model. Selection rule,
matching, splits, MMLU and the factual control are as in notebook 10. The training-free pick is
BES as proposed; the learned eraser is trained with the model frozen and is reported as a
variant, as in notebook 09.

**Part A** reruns notebook 10's four models; **Part B** applies the unchanged protocol to
Qwen3.5-9B and Llama-2-7B-Chat, which v2 was not designed on. Llama-2 here uses the Anthropic
splits of notebooks 02–05 and 10, not Rimsky et al.'s items, so it cross-checks notebooks 07–09
under a common protocol rather than reproducing their numbers. Muse Glimmer 30B is optional and
off by default (`RUN_MUSE = False`).

**How to read it.** `m10b_summary` has notebook 10's columns plus the v2 choices (`bes_layer`,
`bes_window`, `bes_span`, `learned_layer`, `batch_size`, `format_switched`, the `cos_*_vs_caa`
novelty checks) and the no-opinion KL for each method (`*_kl_neutral`); the training-free pick
keeps the `svd_*` column prefix for comparison with notebook 10, and `svd_desc` names its estimator; `m10b_v1_vs_v2` puts the v1 and v2 headline numbers side by side;
`m10b_<model>_format_check` records the format check.

**How it ran.** The first five models ran on a Colab A100; Llama-2-7B ran on a local RTX 5080
(16 GB, WSL2) after the Colab session ended. The notebook holds both runs' outputs in one file:
each model's log comes from the run that produced its results, and its opening cell says which
run each cell's output comes from. The format check switched every chat-format model to the raw
`Answer: (` format (letter mass 0.81–1.00) and, for Llama-2 on the local card, to one prompt at a
time, because batch padding moved its answers by a mean of 0.010 against the 0.01 limit. Scoring
one prompt at a time changes speed only.

**Findings** (test questions; learned erasers against CAA matched to the same validation
sycophancy; *no-opinion KL* is the change on prompts with no opinion in them):

| Model | Eraser | Syco BES / CAA | KL BES / CAA (p) | No-opinion KL BES / CAA | MMLU BES / CAA (none) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | k = 4 | 0.478 / 0.493 | **0.069** / 0.303 (<0.001) | **0.062** / 0.273 | **0.541** / 0.295 (0.595) |
|  | k = 16 | 0.474 / 0.484 | **0.085** / 0.368 (<0.001) | **0.078** / 0.339 | **0.534** / 0.260 (0.595) |
| Qwen3.5-2B | k = 4 | 0.466 / 0.466 | **0.031** / 0.071 (<0.001) | **0.018** / 0.061 | **0.553** / 0.433 (0.600) |
|  | k = 16 | 0.471 / 0.466 | **0.028** / 0.053 (0.007) | **0.018** / 0.048 | **0.557** / 0.413 (0.600) |
| Qwen3.5-9B | k = 4 | **0.440** / 0.467 | **0.092** / 0.195 (<0.001) | **0.067** / 0.189 | **0.702** / 0.556 (0.746) |
|  | k = 16 | **0.438** / 0.464 | **0.093** / 0.197 (<0.001) | **0.076** / 0.193 | **0.682** / 0.512 (0.746) |
| Llama-3.2-3B | k = 4 | 0.509 / 0.514 | 0.230 / 0.247 (0.48) | **0.011** / 0.261 | **0.601** / 0.482 (0.604) |
|  | k = 16 | 0.492 / 0.492 | **0.230** / 0.305 (0.002) | **0.013** / 0.280 | **0.599** / 0.367 (0.604) |
| Llama-3.1-8B | k = 4 | 0.507 / **0.484** | 0.233 / 0.258 (0.44) | **0.133** / 0.299 | 0.554 / 0.520 (0.667) |
|  | k = 16 | 0.487 / 0.484 | 0.323 / 0.257 (0.22) | 0.304 / 0.300 | 0.246 / **0.518** (0.667) |
| Llama-2-7B | k = 4 | 0.518 / **0.499** | **0.308** / 0.384 (0.008) | **0.117** / 0.319 | 0.375 / 0.355 (0.469) |
|  | k = 16 | 0.486 / 0.500 | 0.386 / 0.384 (0.93) | **0.179** / 0.359 | **0.453** / 0.316 (0.469) |

Bold marks a difference with p < 0.05 (paired question-cluster bootstrap; exact McNemar for
MMLU). On the three Qwen models the learned eraser
beats CAA on every measure. On the Llama models KL is lower (Llama-3.2, k = 16) or tied, and the
eraser changes the no-opinion answers far less and keeps more MMLU. Exceptions: on Llama-3.1-8B the k = 16
eraser loses MMLU (0.246 vs 0.518); with k = 4 on Llama-3.1-8B and Llama-2, CAA lands
significantly less sycophantic on test (p = 0.017 and 0.009), so those KL comparisons are not at
equal sycophancy. The training-free pick (centred SVD in every model) ties CAA on KL on four
models, beats it on Llama-2 (0.411 vs 0.593, p = 0.009, CAA less sycophantic) and loses on
Llama-3.1-8B (0.378 vs 0.271, p = 0.003); it costs more MMLU than CAA on both Qwen3.5 models and
less on Llama-3.1-8B. The largest cosine between the BES directions and the CAA vector at the
same layer is 0.17–0.32 per model. The "ignore my opinion" system prompt does not lower sycophancy on
Qwen2.5 (0.738 vs 0.703 with no intervention) and lowers it least of all methods elsewhere.
Caveats: one seed; prompts in the raw format rather than the chat template; the training-free
pick reaches the rank cap on Qwen2.5 and CAA's own-terms pick sits at α = −64 on the Llama
models; validation matching does not guarantee equal test sycophancy.

**What it led to.** The report's §5.7 and Appendix H.

## Running the notebooks

Run 02 before 03, and both before 04 and 05; 01 and 06–10 are independent, and 10b reads notebook 10's summary for its v1-vs-v2 table. Every notebook clones
the repo on Colab, installs requirements, downloads data and mounts Drive; results go to
`results/` and to `MyDrive/w266_bes_v2/`. On an A100, 01–05 take about 1.5 hours, 06 about 10
minutes, 07 about 30, 08 about 60 and 09 about 50; 10 took about 80 minutes for Qwen2.5-1.5B (fp32), 20–25
for Qwen3.5-2B and Llama-3.2-3B and 50 for Llama-3.1-8B; 10b took roughly twice as long per
model (about 5.5 hours for Part A and 1.6 for Qwen3.5-9B on the A100; Llama-2-7B took 4 hours
on a 16 GB RTX 5080, scored one prompt at a time; Muse Glimmer, if switched on, several hours),
saving each model's results as it finishes, so it can be run across several sessions. Notebooks
06–09 need Hugging Face access to `meta-llama/Llama-2-7b-chat-hf`, and notebook 10 to
`meta-llama/Llama-3.2-3B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`, and 10b to those
plus `meta-llama/Llama-2-7b-chat-hf` (a model without access is skipped with a message).

## Design decisions that matter for validity

- **Question-grouped splits.** The 30,051 items contain only 158 distinct questions (NLP 32,
  PhilPapers 109, political 17), each repeated with many biographies. Questions are assigned
  to fit / validation / test (30 / 20 / 50% within each dataset), so no question is shared
  across splits, and items are drawn evenly across datasets and questions.
- **Selection on validation, reporting on test.** Every BES and CAA hyperparameter is chosen on
  validation questions and recorded in `results/selected_config.json`; test questions are
  evaluated once per seed.
- **All answer options are scored.** 27% of PhilPapers items have the user's answer at (C)–(G)
  and 59% offer more than two options. Probabilities are read over each item's own letters.
- **Question-clustered uncertainty.** Items sharing a question are not independent, so
  intervals resample questions (cluster bootstrap). MMLU differences use exact McNemar tests.
- **CAA at full strength.** Rimsky et al.'s recipe (answer-letter position, raw mean difference,
  1,000 fit items) with its own layer, multiplier and span search. CAA's configuration is chosen
  to minimise KL at the sycophancy level BES reaches, the exact point the hypothesis compares.
  Notebook 06 checks the implementation against the published Llama-2 results.
- **Readout.** Prompts end in `(` so the next token is the answer letter. Letter token ids are
  taken in context, which is correct for both byte-level BPE (Qwen) and SentencePiece (Llama).
  Left padding uses explicit position ids. Raw (non-chat) formatting is the default, matching
  the benchmark's `Answer:` format. `Config.chat = True` switches to the chat template (used in notebook 08,
  §12, and notebook 10), but appending `(` after a chat template splits a token these
  tokenizers otherwise merge with the letter, so the answer letters get almost no probability;
  notebook 10b checks this and uses the raw format. The model runs in
  fp32 (Llama-2 in fp16) so small probability differences are not affected by half precision.

## Data

The original data is in `data/` (see `data/README.md` for sources, licences and which notebook uses each file).

| Source | Items | Use |
|---|---|---|
| Anthropic model-written evals, sycophancy (Perez et al., 2023) | 30,051 (158 questions) | fit / validation / test |
| SycophancyEval `answer` (Sharma et al., 2024) | 500 questions (TriviaQA + TruthfulQA) | factual control |
| SycophancyEval `are_you_sure` | 400 questions | pushback transfer |
| MMLU (Hendrycks et al., 2021) | 1,000 random test items | capability |
| Rimsky et al. CAA release | 1,000 + 50, 570 MMLU | implementation benchmark; Llama-2 comparison (06–09) |

## Layout

```
notebooks/                 notebooks 01–09, executed, with outputs; 10, ready to run
src/data.py                opinion/neutral pairs, option letters, question ids
src/bes.py                 prompt formatting, interventions (BES, CAA, LEACE), activation capture, scoring, metrics
src/experiment.py          config, question-grouped splits, Runner (fit/evaluate), auxiliary datasets, persistence
src/stats.py               cluster bootstrap, paired tests, McNemar, AUC, interpolation
src/plots.py               shared figure style (Okabe–Ito palette)
tests/test_mechanics.py    14 offline tests (hooks, masks, padding, LEACE, splits, metrics, stats)
tests/smoke_notebooks.py   runs every notebook on a tiny random model (CPU) to catch code errors
scripts/download_data.sh   re-downloads the raw data (skips files already in data/raw/)
data/                      original datasets (Anthropic sycophancy evals, SycophancyEval, Rimsky et al. release)
results/                   CSVs, figures and selected_config.json from the final runs; preliminary_v0/ holds the first runs
```

## Local checks (no GPU needed)

```bash
pip install -r requirements.txt nbclient ipykernel
bash scripts/download_data.sh && python -m src.data --dataset all
python -m tests.test_mechanics
python -m tests.smoke_notebooks        # about 10 minutes on CPU
```

## Key references

Rimsky et al., ACL 2024 · Perez et al., Findings of ACL 2023 · Sharma et al., ICLR 2024 ·
Arditi et al., NeurIPS 2024 · Belrose et al., NeurIPS 2023 · Zou et al., 2023 (RepE) ·
Bolukbasi et al., NeurIPS 2016 · Wang et al., 2025 (arXiv:2508.02087) · Tan et al., NeurIPS 2024.
