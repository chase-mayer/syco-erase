# Belief-Erasure Steering: Removing the User's Stated Stance from the Residual Stream to Mitigate Sycophancy

**DATASCI 266 — Final Project Proposal (solo)**
Chase Mayer · chase_mayer@berkeley.edu

## Proposal (≈270 words)

Language models change their answers to match opinions that users state about themselves.
Existing inference-time fixes such as Contrastive Activation Addition (CAA) add a fixed
"anti-sycophancy" behaviour vector at every token position, which can overshoot into
contrarianism and provides no principled target for *how much* to steer.

I propose **Belief-Erasure Steering (BES)**, a new algorithm that intervenes on the model's
internal representation of *the user's stated stance* rather than on its output behaviour.
From paired prompts that differ only in the user's self-description, I estimate a low-rank
"stance subspace" of the residual stream (difference-in-means, then a rank-k SVD /
whitened LEACE-style projector) and project activations onto its orthogonal complement at
the biography token positions. The intervention is *invariance-seeking*: the target is that
the model behave as if no opinion had been stated, so over-steering into contrarianism is
impossible by construction.

This yields an unusually crisp objective. The same item with the biography removed is an
oracle, so success is measured as KL divergence between the steered and the no-opinion
answer distribution, alongside sycophancy rate (probability mass on the user-matching
option). Anthropic's model-written sycophancy evaluations (~30k A/B items across the
NLP-survey, PhilPapers and political-typology splits) supply the pairs; SycophancyEval
tests transfer to open-ended and "are you sure?" sycophancy; MMLU measures capability
retention. Baselines: unsteered model, prompting, CAA, behaviour-direction ablation, and
activation patching from the neutral prompt — an oracle upper bound that needs the paired
prompt BES does not. Models are Gemma-2-2B-it and Qwen2.5-7B-Instruct, compared as Pareto
frontiers of sycophancy reduction against capability retention. Implementation is PyTorch
forward hooks; no training is required.

## Why this qualifies as "develop a new NLP algorithm and apply it to a well-known dataset"

The course lists four acceptable project shapes. This is the second one, and deliberately
not the first (applying a known algorithm to new data).

1. **The dataset is fixed and well known.** Anthropic's model-written sycophancy evaluations
   are the same data the CAA paper steered on (Rimsky et al., ACL 2024). Nothing about the
   contribution rests on new data, so the algorithm carries the novelty.
2. **The intervention target is new.** Every published sycophancy intervention I found acts
   on the *output side*: CAA adds a behaviour vector; projection-ablation methods remove a
   *sycophancy* direction; "A Few Bad Neurons" fine-tunes sycophancy-predictive neurons.
   BES acts on the *input side* — the encoded user stance — and leaves the behaviour
   representation untouched.
3. **The objective is new for this task.** Prior methods maximise a behaviour change and
   have no natural stopping point, so strength is tuned against a judge model. BES targets
   an *invariance*: match the model's own no-opinion distribution. That converts an
   open-ended control problem into a well-posed distance-minimisation problem with a
   ground-truth reference, which matters here because the benchmark's opinion questions have
   no factual answer to appeal to.
4. **It is deployable where the closest prior result is not.** Wang et al. (2025) showed by
   activation patching that copying activations from the no-opinion condition reduces
   sycophancy ~36%, but patching requires the paired neutral prompt at inference time and
   the authors explicitly propose no mitigation. BES learns a fixed projector offline and
   needs one forward pass, which is exactly the gap that paper leaves open. Their patching
   result becomes my oracle upper bound.
5. **Falsifiable novelty check, run first.** If the stance subspace turns out to be parallel
   to the CAA behaviour vector (cosine similarity ≈ 1), BES reduces to known work. Measuring
   that similarity is experiment #1 and is a reportable result either way.

## References

1. Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. (2024).
   *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024.
2. Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N.
   (2024). *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024.
3. Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S.
   (2023). *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS 2023.
4. Sharma, M., Tong, M., Korbak, T., et al. (2024). *Towards Understanding Sycophancy in
   Language Models.* ICLR 2024.
5. Tan, D., Chanin, D., Lynch, A., et al. (2024). *Analysing the Generalisation and
   Reliability of Steering Vectors.* NeurIPS 2024.
6. Perez, E., Ringer, S., Lukošiūtė, K., et al. (2023). *Discovering Language Model
   Behaviors with Model-Written Evaluations.* Findings of ACL 2023. (dataset)

Cited as motivation, not counted toward the four required peer-reviewed references:
Wang, K., Li, J., Yang, S., Zhang, Z., & Wang, D. (2025). *When Truth Is Overridden:
Uncovering the Internal Origins of Sycophancy in Large Language Models.* arXiv:2508.02087.

## Planned deliverables

- 4–6 page ACL-format report, ~6 minute presentation, this repository.
- Milestone: EDA on the pair construction + baseline (unsteered and CAA) numbers.
