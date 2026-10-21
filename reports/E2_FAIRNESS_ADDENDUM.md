# E2 fairness addendum and requested controls

Updated 21 October 2026. This supplements the accepted E2 evaluation; no E2 checkpoint, outer test set or hyperparameter was changed. E1 three-seed fitting and CXR layout completed in CPU job 217931. Additional design validation and prior logistic fold comparisons are job 217940.

## Equally sized E1 ensembles

LightGBM comparators were fitted at seeds 0, 1 and 2 on exactly the original five patient partitions, using the original fixed settings, training-only preprocessing/PCA and validation early stopping. Covered arms: context, vitals, labs, CXR, EHR and all modalities, with both count variants where applicable, plus the E1 all-EHR mask-only control. Validation probabilities were averaged before selecting ensemble thresholds. No seed was selected by test performance.

Every one of the 11 arms produced identical probabilities across seeds: maximum patient-level probability difference is 0.0. With no feature or row subsampling, the fixed deterministic LightGBM has no observed seed diversity here. The requested three-model means therefore reproduce the seed-0 scores, including EHR with counts AUROC 0.8241 and the mask-only AUROC 0.6867. This is an equal-sized fit comparison, not three independent sources of uncertainty.

## Paired deltas and outer fold signs

The primary comparison below is E2 three-seed mean minus E1 three-seed LightGBM mean. Each fold delta is calculated on that fold’s held-out patients after averaging seeds. Positive and negative counts describe consistency, not a significance test: outer-fold models have overlapping training data. The patient bootstrap also remains conditional on trained models. AP means average precision.

| Comparison | Delta AUROC with 95% conditional CI | Fold deltas 0 1 2 3 4 | Positive negative zero folds | Approximate resolution | Interpretation |
|---|---|---|---|---|---|
| context versus context no counts | +0.0222 [+0.0126, +0.0315] | +0.0265 / +0.0218 / +0.0219 / +0.0139 / +0.0139 | 5/0/0 | 0.0096 | Conditional CI excludes zero |
| vitals versus vitals no counts | +0.0104 [-0.0023, +0.0252] | +0.0218 / +0.0336 / -0.0120 / +0.0001 / +0.0033 | 4/1/0 | 0.0147 | Unresolved |
| vitals versus vitals with counts | +0.0036 [-0.0074, +0.0169] | -0.0059 / +0.0220 / -0.0004 / -0.0177 / +0.0025 | 2/3/0 | 0.0134 | Unresolved |
| labs versus labs no counts | +0.0240 [+0.0057, +0.0421] | +0.0330 / +0.0027 / +0.0024 / +0.0044 / +0.0465 | 5/0/0 | 0.0183 | Conditional CI excludes zero |
| labs versus labs with counts | -0.0068 [-0.0225, +0.0096] | +0.0176 / -0.0092 / -0.0054 / -0.0183 / -0.0019 | 1/4/0 | 0.0164 | Unresolved |
| cxr versus cxr no counts | +0.0456 [+0.0253, +0.0654] | +0.0625 / +0.0594 / +0.0589 / +0.0407 / +0.0209 | 5/0/0 | 0.0203 | Conditional CI excludes zero |

The count-aware vitals comparison is +0.0036 AUROC, with 2 positive and 3 negative fold deltas. The count-aware labs comparison is −0.0068, with 1 positive and 4 negative deltas. Both are unresolved. Context, no-count labs and CXR improve over their LightGBM references conditionally in all five folds; that does not establish superiority to every baseline or an incremental fusion contribution. No-count vitals is unresolved despite four positive fold deltas.

## Approximate sensitivity

Across these six LightGBM pairs, the conservative 95% interval half-width ranges from 0.0096 to 0.0203 AUROC, with median 0.0156. Thus differences of about 0.01–0.02 are the practical resolution scale for many of these comparisons. There is no single cohort-wide minimum: correlated model predictions can give narrower paired intervals.

The per-pair approximation is the larger distance from the observed delta to either CI endpoint. If the CI includes zero, or the absolute difference is below this approximation, label it unresolved, never equal. A normal-approximation 80%-power design heuristic is about 1.43 times that half-width; the CSV includes it, but it is not a prospective powered sample-size calculation and omits retraining/split/site variability. Tests of equivalence were not performed.

## Prior deterministic logistic reference claims

Logistic references remain labeled as their original deterministic seed-0 fits; the requested three-seed LightGBM ensemble comparisons above are the primary fair references. Fold deltas for the accepted report’s logistic comparisons are retained below.

| Comparison | Delta AUROC and 95% CI | Five outer fold deltas | Positive negative zero | Status |
|---|---|---|---|---|
| context versus context logistic__counts0 | +0.0024 [-0.0023, +0.0072] | -0.0007 / +0.0001 / +0.0061 / +0.0029 / +0.0041 | 4/1/0 | unresolved |
| vitals versus vitals logistic__counts0 | +0.0521 [+0.0359, +0.0683] | +0.0711 / +0.0628 / +0.0468 / +0.0422 / +0.0430 | 5/0/0 | conditional_interval_excludes_zero |
| vitals versus vitals logistic__counts1 | +0.0011 [-0.0117, +0.0149] | -0.0016 / +0.0184 / -0.0246 / -0.0157 / +0.0302 | 2/3/0 | unresolved |
| labs versus labs logistic__counts0 | +0.0318 [+0.0145, +0.0496] | +0.0697 / +0.0099 / +0.0408 / +0.0283 / +0.0392 | 5/0/0 | conditional_interval_excludes_zero |
| labs versus labs logistic__counts1 | +0.0034 [-0.0129, +0.0192] | +0.0375 / -0.0102 / +0.0075 / -0.0043 / +0.0137 | 3/2/0 | unresolved |
| cxr versus cxr logistic__counts0 | +0.0087 [-0.0046, +0.0218] | +0.0143 / +0.0238 / +0.0148 / +0.0277 / -0.0021 | 4/1/0 | unresolved |

Both AUROC and AP fold deltas, confidence intervals, sign counts and approximate resolution values are in E2FairComparisons.csv and E2LogisticFoldComparisons.csv. The reporting rule will also apply to every key E3/E4/E5 comparison.

## Temporal mask controls

Values-only and mask-only vitals/labs controls each use the same two-layer temporal encoder, 128 hidden width, four heads, 48 hours, fixed optimizer settings and validation selection. Inputs have 8 channels for vitals and 11 for labs in either ablation. There are 2 modalities × 2 variants × 3 seeds × 5 folds = 60 new fits. No architecture search is performed.

The controls are submitted as resumable free-QOS GPU job 217932. Until completion, their scores are pending. The final evaluator places all six E2 rows (original, values-only, mask-only for vitals/labs) beside the E1 all-EHR mask-only LightGBM reference, AUROC 0.6867. That E1 reference uses both temporal modalities, so it is contextual evidence rather than a matched single-stream input set.

In every E2 arm, entirely unobserved hours are excluded from masked mean pooling, but imputed tokens remain visible to self-attention. Values-only removes explicit mask channels; it does not remove structural missingness information in pooling or imputation. Mask-only receives observation channels alone, never normalized values. These distinctions are essential when interpreting any ablation difference.

## E4 and optional context follow up

The primary RQ2 test is EHR-only E4a versus EHR-only joint concatenation on context, vitals and labs, using matched E2 initializers, folds and three seeds. Full CXR E4a/E4b are secondary. E4b, all requested E5 retraining/masking/permutation tests and CXR masking curves are retained in E4Design.md. Logistic stacking is dropped. Null results will be reported plainly.

The optional context-only 100-epoch rerun was not launched. The current evidence does not establish that its full 15-fit rerun would cost under five minutes. Some original context fits reached the 50-epoch cap; this limitation is noted without spending extra compute or using test outcomes to change the cap.

Evidence: results/e1_ensembles/summary.csv and e2_paired_comparisons.csv; results/evaluation/e2; code fair_metrics.py, e1_ensembles.py, complete_fairness.py and mask_ablation.py. No patient predictions were exported.
