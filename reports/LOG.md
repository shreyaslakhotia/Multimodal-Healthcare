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
