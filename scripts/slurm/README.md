# Slurm entrypoint catalog

Run one required stage at a time in dependency order. These are reproducible
stages, not a list to paste into a terminal. Check [current jobs](../../docs/STATUS.md)
first. Each entrypoint runs synthetic tests before its main work.

| Stage | Batch file | Prerequisite |
|---|---|---|
| Environment | `setup_environment.sbatch` | SSD cache configuration |
| CPU validation | `check_cpu.sbatch` | Environment installed |
| GPU smoke | `check_gpu.sbatch` | Explicit supported GPU request |
| Acquisition | `acquire_data.sbatch` | Access authorization and storage checks |
| Inspection | `inspect_data.sbatch` | Source manifests/files |
| Cohort | `build_cohort.sbatch` | Inspected dictionaries and metadata |
| QC and folds | `build_features.sbatch` | Candidate cohort and event units |
| E0/E1 EHR | `run_baselines.sbatch` | Final partitions |
| Frozen CXR cache | `extract_cxr_cpu.sbatch` | Resolved images and fixed cohort |
| E1 CXR | `run_cxr_baselines.sbatch` | Frozen cache |
| E1 aggregation | `aggregate_results.sbatch` | Locked E1 OOF files |
| E1 figures | `render_figures.sbatch` | Aggregate tables |
| E7 audit | `audit_reports.sbatch` | Fixed cohort, private text and baseline predictions |
| Source counts | `verify_sources.sbatch` | Required extracted sources |
| E2 training | `train_unimodal.sbatch` | E0/E1 gates and GPU smoke |
| E2 evaluation | `evaluate_e2.sbatch` | Complete E2 fits |
| Fair E1 references/layout | `prepare_fairness.sbatch` | E2 evaluation and E1 cache |
| Design validation | `check_e4_design.sbatch` | Environment and fixed source artifacts |
| Temporal controls | `train_mask_ablation.sbatch` | E2 preprocessing and CPU validation |
| E3 training | `train_fusion.sbatch` | Complete E2 encoders |
| E3 standalone evaluation | `evaluate_fusion.sbatch` | E3 and three-seed E1 references |
| Combined evaluation | `evaluate_requested_controls.sbatch` | E3, controls and fair references complete |

GPU scripts require exactly one model through `--gres=gpu:<model>:1`. The mask
script defaults to free resumable QOS; E3 uses it through the submission flag.
No E4/E5/E6 training script is claimed complete yet. See [E4 design](../../docs/E4_DESIGN.md).

All scripts source SSD cache settings, use private job scratch and log under
`/projects/icumultimodal/reports`. Current fixed paths target this project's
cluster deployment; review paths and quota rules before adapting elsewhere.
