# Report-text audit: corrected version 2

Job 215559 completed in 1:29 after 45 tests passed. Source revision f1122d5.
Version inline_headers_v2 supersedes V1: the first parser recognized only
line-start headers and incorrectly combined inline IMPRESSION with FINDINGS.
V1 artifacts are preserved for provenance; its section scores must not be cited
as correctly separated sections. EHR/CXR model results are unaffected.

## Decision

Text is not eligible for the main benchmark or an extra-input comparison arm.
Its timing inside the first 48 hours cannot be established. All 6,225 records
contain lists of study texts, and 5,395 combine multiple studies. Comparing this
longitudinal information to one selected image cannot isolate modality quality.
The augmented column was not modeled; its provenance remains unverified.

## Corrected predictive audit

Fixed train-only TF-IDF (1-2 grams, minimum document frequency 10, maximum 30,000
features) and L2 logistic regression C=1. Five persisted patient folds; inner
validation maximum-F1 threshold; test predictions collected once per fold.
Absent sections are empty strings. Results below cover the entire cohort,
including patients without a populated section. AUPRC is average precision.
No fold required a featureless/prior fallback in corrected V2.

| Section | Nonempty (%) | AUROC | AUPRC | AUROC minus image (95% paired CI) |
|---|---:|---:|---:|---|
| EXAMINATION | 3.92 | 0.478 | 0.108 | -0.128 (-0.158 to -0.098) |
| INDICATION | 4.72 | 0.484 | 0.110 | -0.122 (-0.150 to -0.092) |
| TECHNIQUE | 1.65 | 0.478 | 0.109 | -0.128 (-0.157 to -0.098) |
| COMPARISON | 4.74 | 0.483 | 0.110 | -0.122 (-0.153 to -0.094) |
| FINDINGS | 94.52 | 0.679 | 0.217 | +0.073 (0.046 to 0.100) |
| IMPRESSION | 96.96 | 0.696 | 0.222 | +0.090 (0.066 to 0.114) |

Comparator: primary CXR-only logistic model, AUROC 0.606, AUPRC 0.162; 1,000
paired patient bootstrap draws conditional on fitted models. FINDINGS AUPRC
improvement is 0.055 (0.032 to 0.083); IMPRESSION is 0.060 (0.040 to 0.086).
These are untimed descriptive comparisons, not clean prospective evidence.
Sparse sections have little coverage; do not treat their near-chance scores as
an assessment of the informational value of populated reports alone.

## Outcome language and associations

The fixed outcome lexicon includes death, deceased, expired, postmortem, brain
death, comfort care, DNR/DNI, withdrawal of care/life support and hospice.
There were zero matches in the parsed sections in both outcome classes.
Original and flagged-term-stripped predictions therefore match. This narrow
negative screen does not establish absence of indirect outcome information,
late observations or source transformation. It does not change the exclusion.

Word/phrase associations use fold-0 inner training patients only, minimum
patient document frequency 10, an empirical pooled Dirichlet prior of fixed
strength 1,000, standardized log odds and global Benjamini-Hochberg correction
across sections. Wordcloud weights are absolute log odds among q < 0.05 terms,
not raw frequency or a measure of modality importance. Aggregate wordclouds
remain on the cluster under reports/figures/report_audit/inline_headers_v2.
No patient text, identifiers or predictions were exported.

Method references: [Monroe, Colaresi and Quinn (2008)](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S1047198700002291)
and [SciPy false discovery control](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.false_discovery_control.html).