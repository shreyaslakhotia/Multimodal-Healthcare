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
