# Revised protocol — 20 October 2026

This document resolves brief inconsistencies using the user's subsequent instructions. It is a protocol, not a claim that experiments have run.

## Evaluation

1. Select and persist the five outer patient-grouped, label-stratified folds using seed 42. Each patient belongs to one outer fold.
2. Within each outer training portion, persist one approximately 15% grouped, stratified validation holdout. Use the same partitions for all comparable model arms and seeds.
3. Fix hyperparameters before outcome-based evaluation. Remove the six-trial search and fold-0 tuning. Do not use any fold's validation outcome to change another fold's model specification.
4. Fit imputation, scaling, PCA and class weights on the inner training portion only. Validation may select the early-stopping checkpoint and operating thresholds. Test data may select neither.
5. Do not refit on training plus validation in this stage: retain the model whose checkpoint and thresholds were selected on validation. This avoids needing a second threshold-selection dataset.
6. Score outer test predictions only after selection. Planned masking and permutation evaluations reuse locked models and are declared sensitivity analyses, not selection criteria.
7. Run three deep seeds (0, 1, 2) on identical patient partitions. Summarize per-seed OOF performance and fold dispersion. For the primary pooled comparison, average the three seed probabilities per patient before patient bootstrap; never treat seed copies as independent patients. Use identical bootstrap patient draws for paired comparisons.
8. Define average precision as the reported AUPRC implementation. Use 1,000 patient bootstrap draws and report paired delta AUROC and delta average precision intervals.
9. Select maximum-F1 thresholds on validation; break ties with the highest threshold. For recall targets 0.80 and 0.90, choose the highest validation threshold meeting the target. Report actual test recall without promising that the validation target generalizes.

## Fixed starting configuration

Deep models retain the brief's stated architecture, batch size 64, maximum 50 epochs, patience 10 on validation AUROC, dropout 0.1, gradient clip 1.0, AdamW weight decay 0.01 and learning rate 0.0001. Record the final configuration hash before scoring. Architecture dimensions follow the actual documented context encoding.

Logistic regression: L2, C=1, maximum 2,000 iterations, train-only median imputation and standardization. LightGBM: learning rate 0.05, 15 leaves, minimum 50 rows per leaf, 1,000 maximum estimators, L2 regularization 1, early stopping patience 50 on validation AUROC. No grid search. Record all library defaults and versions before use.

## Cohort and feature interpretations

The estimand is in-hospital mortality among adults still in ICU and alive at the 48-hour landmark. It does not cover deaths or ICU discharge before that landmark.

Preserve the brief's matched cohort for the main comparisons. EHR + CXR and 'all' are the same arm when EHR includes context, vitals and labs; do not spend compute duplicating it. List the unique leave-one-out and add-one-in arms explicitly.

Context dimension is derived from the category vocabulary, including explicit unknown slots. Raw calendar dates and outcome-derived fields remain outside model inputs.

Mask-only prediction can remain informative after LOS restriction because measurement intensity relates to care and severity. AUROC above 0.65 is a flag requiring explanation, not proof of leakage. Retain the other prespecified stop conditions. A shuffled-label result outside its prespecified band requires investigation rather than repeated reshuffling until it passes.

With untimed CXR, the EHR timing controls do not make the multimodal experiment fully leakage-controlled. Report CXR results as exploratory association. Report audit text must remain outside AI tools; inspect locally on the cluster and export aggregate results only.

GCS component aggregation and unit checks must be documented against the actual dictionaries before data processing. No hidden mapping assumptions.

## Efficiency and budget

Use CPU jobs for setup, data work, canaries, tabular models and report auditing. Extract frozen CXR features once. Use E0/E1 gates before deep training; use the brief's priority order. Keep at most two running and five submitted jobs, with no more than two SSH sessions.

Track stage-specific SU from job accounting against a baseline taken at setup; monthly account usage can include unrelated jobs and resets. Stop at 21,000 stage SU (70% of the 30,000 cap). Do not assume the full experiment ladder fits before measuring early runtime.

No LLM inference, agents, or external AI APIs are part of the benchmark. Development assistance does not receive patient data.


## Accepted follow up amendment

See [requested controls](REQUESTED_CONTROLS.md) and [E4 design](E4_DESIGN.md).
