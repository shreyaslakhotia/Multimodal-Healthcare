# Candidate cohort report — 20 October 2026

Source: `$P/results/cohort/summary.json`, CPU job 215446, revision fdc61cf.
Completed in 3:15 after 28 synthetic tests passed. These counts precede final
valid-value QC and final fold assignment.

| Stage | Stays | Deaths |
|---|---:|---:|
| All linked ICU stays | 73,141 | 8,340 |
| Patient in CXR metadata | 32,089 | 3,454 |
| First ICU stay per patient | 19,003 | 1,852 |
| Adults | 19,003 | 1,852 |
| ICU >=48 h, alive at landmark, consistent outcome metadata | 8,934 | 1,010 |
| Selected image exists and decodes | 6,235 | 708 |
| Valid heart rate in window | 6,232 | 708 |
| At least one numeric listed lab, before final QC | 6,225 | 705 |

Candidate mortality prevalence is 11.33%. S0 has 13,086 stays and 1,226 deaths
(9.37%) before final QC. It is only a demonstration cohort, not the benchmark.

Across all adult first stays before the landmark restriction, 150 lacked frontal
paths and 5,632 had unavailable/ambiguous selected paths. No image decode failures
were observed among resolved selected images. These failure counts refer to the
larger pre-landmark candidate population, not solely the final 6,225 stays.
The deterministic alternative image differs for 5,085 main-cohort candidates.

Two adult first stays had an expiration flag with no hospital death timestamp;
these were excluded from the main landmark cohort. No survival flag with a death
timestamp and no disagreement between recorded hospital-death and dod dates were
found. Date-only dod is never assigned a fictitious midnight landmark timestamp.

The 48-hour compact subsets contain only the named variables, joined by stay ID
for vitals and by patient plus exact stay window for labs. No full event CSV was
loaded into pandas. Event units match expected definitions; detailed item counts
are in `vitals_unitcounts.json` and `labs_unitcounts.json` on the cluster.

CXR remains untimed variant A. After valid-value QC, observation eligibility will
be rechecked, final folds persisted once, and positive counts per fold reported.

## Finalization

Feature/QC job 215456 completed successfully in 15 seconds after 35 tests passed.
No additional stays were removed: the final main cohort is 6,225 stays/705 deaths,
and S0 is 13,086/1,226. Final outer-fold positive counts are 136, 158, 125, 141,
145. Fold assignments and inner partitions are persisted under `$P/data/processed`;
no fold has fewer than 40 positives. No global normalization was fitted.
