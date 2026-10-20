# Source data report — 20 October 2026

## Acquisition and provenance

CPU job 215029 completed successfully in 1:55:32. The nine specified MIMIC-IV
tables total 42,081,445,897 uncompressed bytes. SHA256 digests and verified byte
counts are retained in `$P/results/acquisition`; computed digests do not establish
independent upstream checksum verification. Physical CSV row counts have not
yet been validated. Original backups and the CXR ZIP remain on HDD; only required
IV tables and two CXR metadata CSVs are on SSD.

The Kaggle IV page is labeled MIT, but its LICENSE.txt states PhysioNet
Credentialed Health Data License 1.5.0. The user confirmed required authorization
after reviewing this discrepancy. Preserve the included license as provenance.

## CXR structure

Inspection job 215035 completed successfully. Metadata has 64,586 and 500 rows,
each with a unique patient within its file, and 65,086 patients across both
files. Rows contain image lists grouped by view. Combined image lists contain
371,951 entries, not a verified count of readable images.

Neither CSV has a date/time column. All 200 sampled frontal paths retain original
patient/study structure. User-supplied MIMIC-CXR metadata might permit later
timestamp alignment; no such metadata has been obtained or used.

Of 200 sampled frontal paths, 136 matched unique ZIP-member basenames. None
matched stored paths verbatim. This measures path availability, not decoding.
Each selected image must resolve unambiguously and decode before inclusion.

Variant A remains: first listed AP, otherwise first PA, without falling back to
another image if the selected image is missing. Inputs are untimed and may come
from another admission or after the prediction window. Image-associated gains
or losses are exploratory, not clean temporal evidence.

## Report text

Original and augmented text columns exist. In the first 20 inspected rows per
CSV, strings differed in every case. This does not establish augmentation
method, provenance or timing. No text enters the main benchmark. Section parsing
and outcome-language auditing remain pending; no patient text was exported.

## Item definitions and units

All 15 specified vital IDs and 17 specified lab IDs exist with labels consistent
with the listed concepts. Bicarbonate 50803 is calculated whole-blood bicarbonate;
50882 is chemistry bicarbonate. Document these contexts when merging.

Vital dictionary units are bpm, mmHg, respiratory rate per minute, SpO2 percent,
and Celsius/Fahrenheit; GCS components have no dictionary unit. The lab dictionary
has no unit field. Actual event-unit counts will be produced before features.
Lab-unit verification and physiological QC are not yet complete.

## Pending

Candidate funnel, image decode counts, label consistency, event-unit verification,
valid-value QC, final observation eligibility and final folds.
