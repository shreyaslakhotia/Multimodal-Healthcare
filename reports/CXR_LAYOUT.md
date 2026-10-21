# CXR layout and selection report

Updated 21 October 2026. Aggregate metadata and archive-central-directory inspection only; no PhysioNet metadata was requested or obtained. This is a prerequisite report for the secondary CXR E4 arm.

## Identifiers and timing

All 371,951 listed paths contain a study-directory identifier of the form s followed by digits. All filenames match the structural form of a 40-hex-character DICOM identifier, with optional hyphens. The main cohort also has complete structural matches for all 72,994 listed paths. These are structural identifier checks, not an independent join to canonical PhysioNet metadata. No actual identifiers or patient paths are exported.

Study and DICOM identifiers do not provide acquisition times by themselves. Neither metadata CSV has usable timestamps, and list order has not been established as chronological. First AP means first in the supplied list, not earliest inside the first 48 hours. The user will discuss timestamp metadata with supervisors.

## Images and studies per patient

| Population | Patients | Listed images | Mean images | Median images with IQR | Image count range | Median studies with IQR |
|---|---:|---:|---:|---|---|---|
| All metadata | 65,086 | 371,951 | 5.71 | 3 [2, 6] | 1–174 | 1 [1, 3] |
| Matched cohort | 6,225 | 72,994 | 11.73 | 8 [4, 15] | 1–167 | 5 [3, 11] |

Listed and unique-listed image counts agree in both populations. Counts describe metadata entries, not a claim that every listed archive image is available or decoded. The one selected image per included patient was separately resolved and decoded during cohort construction.

## Alternatives to the selected image

The primary rule selects AP for 5,693/6,225 patients (91.45%) and falls back to PA for 532 (8.55%).

In the matched cohort, 5,837/6,225 (93.77%) have multiple listed images, and 5,395 (86.67%) have multiple studies.

An additional frontal image of either AP or PA view is listed for 5,420/6,225 patients (87.07%). A same-view alternative is listed and resolves in the archive for 5,085/6,225 (81.69%). This is why the previously fixed alternate-image sensitivity selection differs for 5,085 patients.

Among the 5,693 patients whose selected image is AP specifically, 4,835 (84.93%) have another available AP image. This denominator is AP-selected patients, not all patients.

## Interpretation

The matched cohort is enriched for repeated imaging compared with the full metadata population. Available alternatives can span studies and admissions. The existing primary selection is retained to avoid outcome-based image choice. EHR-only concatenation versus EHR-only cross-attention is the primary RQ2 test; CXR is a secondary arm with timing limitations.

Evidence: results/data_quality/cxr_layout.json; main-cohort selection and decode checks in results/cohort; implementation src/icu_mm/data/cxr_layout.py.
