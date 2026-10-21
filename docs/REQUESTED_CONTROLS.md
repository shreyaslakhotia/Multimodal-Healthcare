# Requested controls and revised primary comparison

User-approved amendment on 21 October 2026, following E2 acceptance. No test
predictions are used to change fixed hyperparameters. Logistic stacking is dropped.

- Fit seeds 0/1/2 for fixed E1 LightGBM counterparts, including both count variants,
  and report their actual prediction diversity. Keep folds/preprocessing identical.
- Every key E2/E3/E4/E5 claim includes paired AUROC/AP deltas in all five outer
  folds and sign counts, as well as conditional pooled patient intervals. Fold
  signs are descriptive because training sets overlap. Use pair-specific
  approximate CI half-width as a sensitivity scale; differences below it or
  spanning zero are unresolved, never equal. This is not an equivalence test.
- Values-only/mask-only E2 vitals/labs: 60 fixed new fits, with original observed
  hour pooling retained. Values-only has no explicit mask channels but still
  carries structural missingness. Self-attention still sees imputed hourly tokens.
- CXR layout must be documented before E4. No PhysioNet metadata is acquired.
  See reports/CXR_LAYOUT.md for study/DICOM identifier shape checks and alternatives.
- Main RQ2: EHR-only E4a versus EHR-only joint concatenation. Full CXR E4 is
  secondary. E4b and every E5 retraining/masking/permutation/CXR-curve task remain.
  See docs/E4_DESIGN.md; no null result is suppressed or called equivalence.
- Do not spend on the optional context 100-epoch cap unless the full rerun is
  known to fit under five minutes. It was not run; some 50-epoch fits hit the cap.

Completed CPU jobs: 217931 (fair ensembles/layout, 1:27) and 217940 (refined
layout, logistic fold deltas, E4 architecture tests, 0:41). Latest suite: 54 passed.

Current GPU chain: one E3 job 217883 and one mask job 217932, both one A6000,
free override-limits-but-killable QOS. They checkpoint each epoch and resume on
requeue. CPU evaluator 217933 waits for both; it produces E3_RESULTS.md and
MASK_CONTROLS.md plus full aggregate tables. Maximum two running jobs are maintained.
The mask job was briefly held while the short design CPU job ran, then released.

217740, 217849, 217853 and 217854 are cancelled, all before allocation. No record
for 217741 was found under this user over the project date range; it is not in
squeue. Do not claim its cancellation without an accounting record. No duplicate
E3 copy is active. Normal charged stage GPU use is 64.2 SU so far. Free-QOS work
uses no charged SU under the documented cluster policy; retain allocated runtime
in accounting separately rather than pretending the GPU work consumed no resources.