import csv,json,statistics
from pathlib import Path

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--input-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
SOURCE=args.input_dir; OUT=args.output_dir
layout=json.loads((SOURCE/'cxr_layout.json').read_text())
def read(name): return list(csv.DictReader((SOURCE/name).open(newline='')))
fair=read('e2_paired_comparisons.csv')
legacy=read('e2_logistic_fold_comparisons.csv') if (SOURCE/'e2_logistic_fold_comparisons.csv').exists() else []
summary=read('summary.csv')
if len(summary)!=11 or any(float(row['seed_max_probability_difference'])!=0 for row in summary):
    raise ValueError('These checkpoint narratives require the verified 11 deterministic E1 arms; review prose before reporting a different result')
def pct(n,d): return f'{100*n/d:.2f}%'
def fmt(x): return f'{float(x):+.4f}'
def ci(row,metric='auroc'):
    return f"{fmt(row[metric+'_delta'])} [{fmt(row[metric+'_lower'])}, {fmt(row[metric+'_upper'])}]"

population=layout['populations']; full=population['all_metadata']; main=population['main_cohort']; counts=main['counts']; n=main['patients']
cxr=['# CXR layout and selection report','Updated 21 October 2026. Aggregate metadata and archive-central-directory inspection only; no PhysioNet metadata was requested or obtained. This is a prerequisite report for the secondary CXR E4 arm.','## Identifiers and timing',
f"All {full['counts']['image_paths']:,} listed paths contain a study-directory identifier of the form s followed by digits. All filenames match the structural form of a 40-hex-character DICOM identifier, with optional hyphens. The main cohort also has complete structural matches for all {counts['image_paths']:,} listed paths. These are structural identifier checks, not an independent join to canonical PhysioNet metadata. No actual identifiers or patient paths are exported.",
'Study and DICOM identifiers do not provide acquisition times by themselves. Neither metadata CSV has usable timestamps, and list order has not been established as chronological. First AP means first in the supplied list, not earliest inside the first 48 hours. The user will discuss timestamp metadata with supervisors.',
'## Images and studies per patient',
'| Population | Patients | Listed images | Mean images | Median images with IQR | Image count range | Median studies with IQR |',
'|---|---:|---:|---:|---|---|---|']
for label,data in [('All metadata',full),('Matched cohort',main)]:
    a=data['listed_images_per_patient']; s=data['studies_per_patient']
    cxr.append(f"| {label} | {data['patients']:,} | {a['total']:,} | {a['mean']:.2f} | {a['median']:.0f} [{a['q25']:.0f}, {a['q75']:.0f}] | {a['min']:.0f}–{a['max']:.0f} | {s['median']:.0f} [{s['q25']:.0f}, {s['q75']:.0f}] |")
cxr += ['Listed and unique-listed image counts agree in both populations. Counts describe metadata entries, not a claim that every listed archive image is available or decoded. The one selected image per included patient was separately resolved and decoded during cohort construction.',
'## Alternatives to the selected image',
f"The primary rule selects AP for {counts['selected_ap']:,}/{n:,} patients ({pct(counts['selected_ap'],n)}) and falls back to PA for {counts['selected_pa']:,} ({pct(counts['selected_pa'],n)}).",
f"In the matched cohort, {counts['patients_with_multiple_images']:,}/{n:,} ({pct(counts['patients_with_multiple_images'],n)}) have multiple listed images, and {counts['patients_with_multiple_studies']:,} ({pct(counts['patients_with_multiple_studies'],n)}) have multiple studies.",
f"An additional frontal image of either AP or PA view is listed for {counts['patients_with_additional_frontal_image']:,}/{n:,} patients ({pct(counts['patients_with_additional_frontal_image'],n)}). A same-view alternative is listed and resolves in the archive for {counts['patients_with_same_view_available_alternative']:,}/{n:,} ({pct(counts['patients_with_same_view_available_alternative'],n)}). This is why the previously fixed alternate-image sensitivity selection differs for 5,085 patients."]
if 'ap_selected_patients_with_ap_available_alternative' in counts:
    a=counts['ap_selected_patients_with_ap_available_alternative']; d=counts['selected_ap']
    cxr.append(f"Among the {d:,} patients whose selected image is AP specifically, {a:,} ({pct(a,d)}) have another available AP image. This denominator is AP-selected patients, not all patients.")
cxr += ['## Interpretation','The matched cohort is enriched for repeated imaging compared with the full metadata population. Available alternatives can span studies and admissions. The existing primary selection is retained to avoid outcome-based image choice. EHR-only concatenation versus EHR-only cross-attention is the primary RQ2 test; CXR is a secondary arm with timing limitations.','Evidence: results/data_quality/cxr_layout.json; main-cohort selection and decode checks in results/cohort; implementation src/icu_mm/data/cxr_layout.py.']
(OUT/'CXRLayout.md').write_text(('\n\n'.join(cxr)+'\n').replace('|\n\n|','|\n|'),encoding='utf-8')

widths=[float(row['auroc_resolution_95_approx']) for row in fair]
lines=['# E2 fairness addendum and requested controls','Updated 21 October 2026. This supplements the accepted E2 evaluation; no E2 checkpoint, outer test set or hyperparameter was changed. E1 three-seed fitting and CXR layout completed in CPU job 217931. Additional design validation and prior logistic fold comparisons are job 217940.',
'## Equally sized E1 ensembles',
'LightGBM comparators were fitted at seeds 0, 1 and 2 on exactly the original five patient partitions, using the original fixed settings, training-only preprocessing/PCA and validation early stopping. Covered arms: context, vitals, labs, CXR, EHR and all modalities, with both count variants where applicable, plus the E1 all-EHR mask-only control. Validation probabilities were averaged before selecting ensemble thresholds. No seed was selected by test performance.',
'Every one of the 11 arms produced identical probabilities across seeds: maximum patient-level probability difference is 0.0. With no feature or row subsampling, the fixed deterministic LightGBM has no observed seed diversity here. The requested three-model means therefore reproduce the seed-0 scores, including EHR with counts AUROC 0.8241 and the mask-only AUROC 0.6867. This is an equal-sized fit comparison, not three independent sources of uncertainty.',
'## Paired deltas and outer fold signs',
'The primary comparison below is E2 three-seed mean minus E1 three-seed LightGBM mean. Each fold delta is calculated on that fold’s held-out patients after averaging seeds. Positive and negative counts describe consistency, not a significance test: outer-fold models have overlapping training data. The patient bootstrap also remains conditional on trained models. AP means average precision.',
'| Comparison | Delta AUROC with 95% conditional CI | Fold deltas 0 1 2 3 4 | Positive negative zero folds | Approximate resolution | Interpretation |',
'|---|---|---|---|---|---|']
for row in fair:
    name=row['comparison'].replace('e2_','').replace('_minus_e1_lightgbm_',' versus ').replace('__counts0',' no counts').replace('__counts1',' with counts')
    delta=' / '.join(fmt(row[f'auroc_fold{k}_delta']) for k in range(5))
    signs='/'.join(row[f'auroc_{s}_folds'] for s in ('positive','negative','zero'))
    status='Unresolved' if row['auroc_status']=='unresolved' else 'Conditional CI excludes zero'
    lines.append(f"| {name} | {ci(row)} | {delta} | {signs} | {float(row['auroc_resolution_95_approx']):.4f} | {status} |")
lines += ['The count-aware vitals comparison is +0.0036 AUROC, with 2 positive and 3 negative fold deltas. The count-aware labs comparison is −0.0068, with 1 positive and 4 negative deltas. Both are unresolved. Context, no-count labs and CXR improve over their LightGBM references conditionally in all five folds; that does not establish superiority to every baseline or an incremental fusion contribution. No-count vitals is unresolved despite four positive fold deltas.',
'## Approximate sensitivity',
f"Across these six LightGBM pairs, the conservative 95% interval half-width ranges from {min(widths):.4f} to {max(widths):.4f} AUROC, with median {statistics.median(widths):.4f}. Thus differences of about 0.01–0.02 are the practical resolution scale for many of these comparisons. There is no single cohort-wide minimum: correlated model predictions can give narrower paired intervals.",
'The per-pair approximation is the larger distance from the observed delta to either CI endpoint. If the CI includes zero, or the absolute difference is below this approximation, label it unresolved, never equal. A normal-approximation 80%-power design heuristic is about 1.43 times that half-width; the CSV includes it, but it is not a prospective powered sample-size calculation and omits retraining/split/site variability. Tests of equivalence were not performed.',
'## Prior deterministic logistic reference claims',
'Logistic references remain labeled as their original deterministic seed-0 fits; the requested three-seed LightGBM ensemble comparisons above are the primary fair references. Fold deltas for the accepted report’s logistic comparisons are retained below.']
if legacy:
    lines += ['| Comparison | Delta AUROC and 95% CI | Five outer fold deltas | Positive negative zero | Status |','|---|---|---|---|---|']
    for row in legacy:
        name=row['comparison'].replace('e2_','').replace('_minus_main__',' versus ').replace('__logistic',' logistic').replace('__selectionmain','')
        delta=' / '.join(fmt(row[f'auroc_fold{k}_delta']) for k in range(5))
        signs='/'.join(row[f'auroc_{s}_folds'] for s in ('positive','negative','zero'))
        lines.append(f"| {name} | {ci(row)} | {delta} | {signs} | {row['auroc_status']} |")
lines += ['Both AUROC and AP fold deltas, confidence intervals, sign counts and approximate resolution values are in E2FairComparisons.csv and E2LogisticFoldComparisons.csv. The reporting rule will also apply to every key E3/E4/E5 comparison.',
'## Temporal mask controls',
'Values-only and mask-only vitals/labs controls each use the same two-layer temporal encoder, 128 hidden width, four heads, 48 hours, fixed optimizer settings and validation selection. Inputs have 8 channels for vitals and 11 for labs in either ablation. There are 2 modalities × 2 variants × 3 seeds × 5 folds = 60 new fits. No architecture search is performed.',
'The controls are submitted as resumable free-QOS GPU job 217932. Until completion, their scores are pending. The final evaluator places all six E2 rows (original, values-only, mask-only for vitals/labs) beside the E1 all-EHR mask-only LightGBM reference, AUROC 0.6867. That E1 reference uses both temporal modalities, so it is contextual evidence rather than a matched single-stream input set.',
'In every E2 arm, entirely unobserved hours are excluded from masked mean pooling, but imputed tokens remain visible to self-attention. Values-only removes explicit mask channels; it does not remove structural missingness information in pooling or imputation. Mask-only receives observation channels alone, never normalized values. These distinctions are essential when interpreting any ablation difference.',
'## E4 and optional context follow up',
'The primary RQ2 test is EHR-only E4a versus EHR-only joint concatenation on context, vitals and labs, using matched E2 initializers, folds and three seeds. Full CXR E4a/E4b are secondary. E4b, all requested E5 retraining/masking/permutation tests and CXR masking curves are retained in E4Design.md. Logistic stacking is dropped. Null results will be reported plainly.',
'The optional context-only 100-epoch rerun was not launched. The current evidence does not establish that its full 15-fit rerun would cost under five minutes. Some original context fits reached the 50-epoch cap; this limitation is noted without spending extra compute or using test outcomes to change the cap.',
'Evidence: results/e1_ensembles/summary.csv and e2_paired_comparisons.csv; results/evaluation/e2; code fair_metrics.py, e1_ensembles.py, complete_fairness.py and mask_ablation.py. No patient predictions were exported.']
(OUT/'E2FairnessAddendum.md').write_text(('\n\n'.join(lines)+'\n').replace('|\n\n|','|\n|'),encoding='utf-8')
for source,dest in [('e2_paired_comparisons.csv','E2FairComparisons.csv'),('e2_logistic_fold_comparisons.csv','E2LogisticFoldComparisons.csv'),('summary.csv','E1ThreeSeedResults.csv')]:
    if (SOURCE/source).exists(): (OUT/dest).write_bytes((SOURCE/source).read_bytes())
print('AGGREGATE_REPORTS_CREATED',min(widths),max(widths),statistics.median(widths))
