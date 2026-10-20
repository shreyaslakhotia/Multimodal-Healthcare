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
