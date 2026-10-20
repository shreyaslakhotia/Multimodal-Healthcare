# Source data report

Updated 20 October 2026. Nine MIMIC-IV tables total 42,081,445,897 uncompressed
bytes. Acquisition job 215029 completed in 1:55:32. SHA256 digests and byte
counts are recorded; these are computed locally, not independently checked
upstream digests. Original ZIP backups and the CXR ZIP remain on HDD. Only
required tables, metadata and derived training artifacts are on SSD.

The Kaggle IV page says MIT, but the included license is PhysioNet Credentialed
Health Data License 1.5.0. The user confirmed required access authorization after
reviewing that discrepancy. The included license remains the provenance record.

## Verified source counts

CPU job 215533 counted data rows, excluding headers.

| Table | Rows |
|---|---:|
| admissions | 431,088 |
| patients | 299,777 |
| labevents | 118,057,948 |
| d_labitems | 1,623 |
| diagnoses_icd | 4,752,265 |
| d_icd_diagnoses | 109,775 |
| icustays | 73,141 |
| chartevents | 314,035,266 |
| d_items | 4,014 |

## CXR and text structure

Two metadata CSVs contain 64,586 and 500 distinct patients, 65,086 combined,
and 371,951 image-list entries. List entries are not a verified image count.
Neither CSV has timestamps. The primary choice is first listed AP, otherwise
first PA; no fallback if that selected file is unavailable. Final main images
resolve uniquely, decode successfully, and match their source patient identity
(6,225 checked; zero mismatches or unparsed identities). The alternate same-view
selection differs for 5,085 patients. Frozen feature extraction cached 11,310
unique primary/alternate images for reuse without repeated archive decoding.

All 6,225 original text values are list encoded. Text-entry count matches study
count in every patient; 5,395 have multiple studies and 5,389 have multiple
distinct text entries. Thus text audit combines longitudinal studies rather than
one contemporaneous report per selected image. Timing and original/augmented
text provenance are unverified. Text is excluded from the main benchmark.
A line-anchored parser missed inline headers in audit version one; that version
is preserved, superseded by the explicit inline_headers_v2 audit namespace.
See REPORT_AUDIT.md for corrected results. No patient text was exported.

## Cohort, units and QC

Final main cohort: 6,225 first adult ICU stays, 705 deaths; S0 diagnostic cohort:
13,086 stays, 1,226 deaths. Main eligibility requires survival and ICU residence
at 48 hours and valid HR plus numeric lab observations. See COHORT_REPORT.md.
All actual vital/lab event units match intended units. Broad fixed artifact
bounds are in configs/features.yaml; final QC removed no further eligible stays.
GCS requires all three components within the same hour; ETT verbal responses
are unassessable/missing. Bicarbonate 50803 is calculated whole-blood bicarbonate,
whereas 50882 is chemistry bicarbonate. Train-only imputation, normalization,
vocabulary and PCA are applied within persisted partitions.

EHR timing is controlled to [intime, intime + 48 hours). CXR and report timing
cannot be established; their exploratory associations cannot establish temporally
clean clinical benefit. No timed metadata has been acquired.