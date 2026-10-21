# Decision log

## 20 October 2026 — initial setup

- User selected fixed hyperparameters and authorized efficient correction of protocol contradictions.
- Inner validation may select checkpoints and thresholds; fitting preprocessing and class weights remains training-only.
- Context width will follow the documented category encoding, including unknown slots.
- User says cluster folder names omit hyphens and underscores. Expected project links are `/projects/icumultimodal` and `/projects/icumultimodalarchive`; neither is verified yet. This naming change does not alter command flags, Python names or cluster-managed `_ssd` and `_hdd` paths.
- User has not completed cache configuration or other setup. Do not assume an environment or dataset exists.
- Reviewed supplied brief, diagram, official cluster rules, MedFuse and AgentRx. Kaggle HTML did not expose file layout.
- SSH key-only test reached login-1 but failed authentication. No password attempted; no remote inspection, writes or jobs performed.
- Created local repository scaffold. No patient data downloaded or inspected.
- User believes the Kaggle copies are freely usable without PhysioNet credentialing. Dataset-specific provenance and reuse terms remain to be checked; do not describe this belief as verified licensing.
- Infrastructure and synthetic-data development can continue independently of dataset access clarification.

## 20 October 2026 — access verified and environment job submitted

- Dedicated SSH key authentication works. Explicit login-shell execution initializes modules correctly; noninteractive commands must not assume this environment.
- User verified project allowances: SSD 100 GB assigned out of 150 GB account allowance; HDD 200 GB assigned out of 300 GB. Plan against assigned limits.
- Project target permissions are 2700 (owner-only access plus setgid). Earlier 777 output was the symlink mode, not the directory mode; corrected checks use stat -L.
- CPU job 215010 submitted from committed revision 8806ff4, 8 CPUs, 1-hour limit, account ug-proj, QoS ug. It passed compute-node reachability checks for Kaggle, PyPI, PyTorch and the weights host. No GPU SU consumed by this job.
- Fixed PyTorch 2.8.0 and torchvision 0.23.0 CUDA 12.8 wheels, as documented on the official previous-versions page. Direct scientific dependencies pinned; resolved versions will be recorded after installation.
- Kaggle public API metadata reports 67,376,460,708 bytes for the full MIMIC-IV source; nine required tables total 42,081,445,897 bytes. MIMIC-CXR reports 17,935,573,408 bytes. Only metadata and the 2,518-byte license have been retrieved locally; no patient data downloaded.
- Uploader metadata labels IV MIT and CXR CC0, but the IV dataset's own mimic-iv-2.1/LICENSE.txt is the PhysioNet Credentialed Health Data License 1.5.0. Treat the metadata labels as conflicting, not proof of unrestricted reuse. Patient-data downloads on hold for clarification. Continue infrastructure and synthetic testing.
- Corrected an anticipated array-shape assertion in the forward-fill test before its execution. New window, forbidden-input and landmark guard tests added; these have not yet run.

## 20 October 2026 — synthetic checks passed; data access confirmed

- User explicitly confirmed required access authorization after being shown the archive-license conflict. This resolves the patient-data access gate; preserve the source-license provenance caveat in documentation.
- Environment setup job 215010 installed packages but failed on the old forward-fill test assertion (shape mismatch in expected array). This was a test bug, not a data canary failure. Corrected the expected-array shape and ran the expanded suite in CPU job 215024.
- Job 215024 completed successfully: 22 synthetic tests passed; pip check reported no broken requirements. Torch 2.8.0+cu128, torchvision 0.23.0+cu128, numpy 1.26.4 and sklearn 1.6.1 verified. GPU behavior remains to be checked separately.
- Measured on CPU allocation: environment 8,268,767,768 bytes; pip cache 1,606,270,765 bytes; conda package cache 940,932,212 bytes. Conservative acquisition plan (all nine IV tables, entire uncompressed CXR size, 8 GB future artifacts/scratch, and 1 GB code/cache-growth reserve) is about 79.8 GB, below the 90 GB SSD stop point.
- Prepared selective resumable acquisition with fixed paths, transport validation, SHA256 recording and conservative size guards. No recursive scans of dataset or image directories. CXR original ZIP stays on HDD; only CSV metadata initially extracted to SSD.

## 20 October 2026 — jobs queued

- Data acquisition job 215029 accepted: CPU-only, 8 CPUs, 4-hour limit; tests gate downloads. L40 GPU smoke job 215030 accepted: one GPU, 5-minute limit, at most 20 SU. Live L40 mixed-state node had all four GPUs allocated, so no immediate GPU capacity was assumed.
- Discovery job 215035 accepted with afterok:215029 and kill-on-invalid-dep enabled, CPU-only, 8 CPUs, 30-minute limit. It checks dictionaries and metadata structure and exports aggregate diagnostics only.
- Latest synchronized code revision is 5b56cc1. All three jobs were pending at last snapshot: acquisition/smoke for Priority, discovery for Dependency. No patient data download or GPU execution had started.
- Additional acquisition and safe-path parser tests are committed but not yet run; the previous successful suite count remains 22. Do not report new tests as passed until the queued CPU job executes them.

## 20 October 2026 — acquisition/inspection complete and repository cleanup

- Acquisition 215029 completed successfully in 1:55:32. All nine required IV tables and two CXR metadata CSVs verified; 25 tests passed.
- Inspection 215035 completed successfully in 15 seconds; 25 tests passed. Metadata has 65,086 distinct patients, no timing columns, and patient-level rows containing image lists. 136/200 sampled image paths matched unique archive-member basenames. All 200 retained original study structure. Continue variant A; no timestamp metadata acquired.
- GPU smoke 215030 still pending for Priority at 19:48 Singapore time. No GPU capacity assumed available.
- User requested cleanup of obsolete exploratory/setup code. Removed completed Windows preflight and duplicate cache helpers; moved reproducible Slurm stages to scripts/slurm and data modules to src/icu_mm/data. Kept acquisition/inspection for reproducibility and synthetic tests for leakage controls. No raw data, environments, archives, or results deleted.
- Updated README, runbook and DATA_REPORT. Removed the obsolete tracked BLOCKED.md because no active blocker exists. The historical setup assertion failure remains recorded above.
- Candidate cohort builder added: deterministic first-stay/adult filters, strict first-AP/otherwise-PA selection, image decode verification, alive-at-48h and LOS constraints, label consistency counts, early-filtered DuckDB event scans, resumable compact subsets, and actual unit-count diagnostics. Final folds are deferred until valid-value QC and final observation eligibility. New cohort tests pending execution.
- Conservative handling of inconsistent outcome metadata: flag 1 with missing hospital deathtime, or flag 0 with a deathtime, is excluded from the main landmark cohort and counted. S0 omits the landmark filter as specified. Date-only dod is cross-checked by date and never assigned a fictitious time for the landmark filter.

## Connectivity interruption after cleanup

- Local cleanup and candidate cohort code committed as dd44d44. Git whitespace checks passed; stale import/script references were removed from current entrypoints and documentation. The new cohort tests have not run.
- Two SCP attempts to the original cluster alias timed out after the earlier successful status read. No alternate hosts or access workarounds attempted. The bundle is saved locally, but the remote checkout still has its previous organization; no new cohort job submitted.
- Asked the user whether the laptop remains on NTUSECURE/NTU VPN. Preserve completed datasets and existing Slurm jobs. Resume by rechecking SSH, syncing committed source, then running the expanded suite on CPU before cohort construction.

## Connectivity restored

- User reconnected; SSH verified successfully. The remote checkout was clean at 5b56cc1, and GPU smoke 215030 remained pending. Resume synchronization and CPU validation before candidate cohort construction.

## Candidate cohort complete and feature QC prepared

- Cleaned source synced at fdc61cf. Candidate cohort CPU job 215446 completed in 3:15 after 28 tests passed. Main candidates: 6,225 stays/705 deaths; S0: 13,086 stays/1,226 deaths. Both exceed the support stop thresholds.
- Actual event-unit counts inspected as aggregate diagnostics. All vital/lab units match intended measurement units; no unexpected unit was observed. Compact filtered events are about 10.5 MB vitals and 1.8 MB labs.
- Broad artifact ranges fixed in configs/features.yaml before performance evaluation. These are researcher-defined QC bounds, not clinical decision thresholds. Retain per-channel drop counts and recheck inclusion after QC.
- GCS ETT verbal response is treated as unassessable/missing, not a score of one. Use valid recorded numeric components (or leading coded-text score if absent), hourly component means, and require all three components within the same hour. No cross-hour component assembly.
- Added fixed LightGBM (AUROC early stopping), train-only context vocabularies with explicit unknown slots, hourly aggregation/summary functions, and tests locally in be94fb2. New code has not yet been tested or synced while the cohort job was running. Feature/fold builder and QC tests now prepared for the next CPU validation.
- Final main folds will be fixed only after QC. S0 has separately persisted seed-42 grouped stratified partitions solely for its diagnostic comparison. No global normalizer, imputer or PCA fit is performed during feature construction.

## Final CPU checkpoint and GPU queue

- Feature QC/folds 215456 completed; 35 tests. EHR baselines 215460 completed
  in 2:39; 36 tests. Frozen CXR features 215471 completed in 3:59; 39 tests.
  CXR baselines 215480 completed in 54 seconds; 40 tests. Aggregation 215486
  completed in 2:27 and plots 215492 in four seconds.
- Final cohort 6,225/705 deaths; 48 tabular configurations, 44 paired contrasts.
  No statistically supported CXR gain; all full-minus-EHR AUROC/AUPRC intervals
  include zero. Image timing remains unknown. Final mask-only LightGBM 0.687
  exceeds the warning threshold; no prespecified hard stop was triggered.
- E2 CPU validation 215519 passed 44 tests and dependency checks. Source row
  counts/report container check 215533 completed in 22 seconds. E2 job 215538
  is queued after GPU smoke 215030; no deep results or GPU SU consumed yet.
- Report audit V1 215500 used a defective line-anchored section parser. The
  independent structure check revealed inline headers. Corrected parser tests
  and V2 source committed f1122d5; V2 job 215559 passed 45 tests and completed
  in 1:29. Preserve V1, cite only V2 section results. FINDINGS AUROC 0.679,
  IMPRESSION 0.696; unknown timing/multiple studies exclude text from main use.
- Aggregate tables, plots, checkpoint, canary and source/audit reports are now
  organized under reports/. Patient data, predictions and checkpoints remain
  private cluster artifacts outside Git. No speculative exploratory scripts
  were retained; reproducible acquisition and inspection stages are retained.
## 21 October 2026 E2 evaluation and E3 submission

- GPU smoke 215030 completed 00:00:19 on L40S, GPU_SMOKE_PASSED; bf16 available.
  E2 training 215538 completed 00:15:44, 60 cases and 12 OOF artifacts confirmed.
  Completion signature e1ed74ed723d556a3dc05899eba2b92423e2fa3162377bc3dd25e1bf9fcc0c23;
  source 0f6bd8d. Actual smoke plus E2 GPU usage is approximately 64.2 SU.
- User requested an E2 evaluation document before the next experiment. CPU
  evaluation 217724 (27f9b56) passed 46 tests and finished in 1:22. No seed
  selection: average three seed probabilities per patient and select thresholds
  using matching averaged validation predictions. Patient intervals and paired
  differences use 1,000 resamples conditional on fitted models.
- Ensemble AUROC: context .6837, vitals .7732, labs .7447, CXR .6145. Against
  count-aware E1 baselines, temporal encoder AUROC differences are unresolved;
  CXR is unresolved against logistic regression. No architecture sweep justified.
  Temporal attention sees imputed hours while pooling excludes entirely unobserved
  hours. Document this implementation detail and preserve missingness warning.
- Eight-page E2 PDF and editable Markdown, four figures and aggregate tables
  committed. The standard DOCX renderer lacked LibreOffice; native Word COM was
  unavailable. Deliver a directly generated and visually checked PDF, preserving
  the unrendered DOCX only as an internal content intermediate. No patient data
  used for local authoring. Inspect all eight rendered pages before delivery.
- E3 early/joint implementation: same E2 fold/seed initializers, deterministic
  frozen embedding cache, frozen encoder eval mode for early fusion, same MLP
  and fixed optimizer settings, checkpoints and signature locks. Late average
  included, stacking and optional LSTM deferred per priority. CPU job 217739
  passed 49 tests and pip check in 11 seconds.
- Submitted one-L40 E3 job 217740, afterok 217739, two-hour maximum (480 SU).
  SSH stalled after submission; read-only squeue/sacct and registry check confirmed
  acceptance and no duplicate was submitted. It is pending Priority. E3 CPU
  aggregation/paired evaluation prepared; no E3 scores exist yet.
## E3 queue update

The pending L40 training 217740 and evaluator 217849 were cancelled before any
GPU allocation after its start estimate slipped to 24 October. Live metadata
showed one free A6000 GPU (9/10 allocated), a supported x86 Ampere alternative.
An initial replacement could not resolve the old completed CPU prerequisite
because it had aged out of Slurm's controller. Accounting still confirms its
49-test success. A fresh identical CPU gate preserves the dependency requirement;
no resource/QoS policy was changed or bypassed.

Current chain: CPU check 217852 -> A6000 E3 217853 -> CPU evaluation 217854.
E3 requests one A6000 for two hours, at most 360 SU; scientific settings are
unchanged. Use these IDs rather than the superseded L40 IDs:

```bash
squeue --me
squeue --start -j 217853
sacct -j 217852,217853,217854 --format=JobID,JobName,State,Elapsed,ExitCode
tail -f /projects/icumultimodal/reports/icue3-217853.out
```

CPU evaluation runs only after successful training. Review results/evaluation/e3
and reports/icue3eval-217854.out; an empty queue alone is not proof of success.

## User accepted E2 and requested controls

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