"""Report original, values-only and mask-only E2 beside E1 missingness control."""
import json
import os
from pathlib import Path
import pandas as pd
from .evaluate_e2 import average_seeds
from .metrics import score,select_threshold
from .bootstrap import patient_intervals
from .fair_metrics import paired_evidence

P=Path('/projects/icumultimodal')


def main():
    if 'SLURM_JOB_ID' not in os.environ: raise RuntimeError('CPU allocation required')
    root=P/'results/deep/e2_mask_ablation'
    if not json.loads((root/'complete.json').read_text())['complete']: raise RuntimeError('Mask controls incomplete')
    out=P/'results/evaluation/mask_controls'; out.mkdir(parents=True,exist_ok=True)
    rows=[]; comparisons=[]
    for modality in ('vitals','labs'):
        original=pd.read_parquet(P/f'results/evaluation/e2/{modality}_ensemble.parquet')
        models={'values_and_masks':original}
        for variant in ('values_only','mask_only'):
            seeds=[pd.read_parquet(P/f'results/oof/e2control__{variant}__{modality}__seed{seed}.parquet') for seed in range(3)]
            ensemble=average_seeds(seeds)
            for fold in range(5):
                validation=average_seeds([pd.read_parquet(root/variant/modality/f'fold{fold}'/f'seed{seed}'/'validation.parquet') for seed in range(3)])
                ensemble.loc[ensemble.fold==fold,'threshold']=select_threshold(validation.y_true,validation.y_prob)
            models[variant]=ensemble
            ensemble.to_parquet(out/f'{modality}_{variant}_ensemble.parquet',index=False)
        for name,model in models.items():
            intervals=patient_intervals(model.y_true,model.subject_id,model.y_prob)
            record=dict(modality=modality,variant=name,**score(model.y_true,model.y_prob,model.threshold))
            for metric in ('auroc','auprc'):
                for bound in ('lower','upper'): record[f'{metric}_{bound}']=intervals[metric][bound]
            rows.append(record)
        for first,second in [('values_and_masks','values_only'),('values_and_masks','mask_only'),('values_only','mask_only')]:
            comparisons.append(dict(modality=modality,comparison=f'{first}_minus_{second}',**paired_evidence(models[first],models[second])))
        baseline=pd.read_parquet(P/'results/e1_ensembles/mask__counts1/ensemble.parquet')
        comparisons.append(dict(modality=modality,comparison='e2_mask_only_minus_e1_all_ehr_mask_3seed',**paired_evidence(models['mask_only'],baseline)))
    baseline=pd.read_parquet(P/'results/e1_ensembles/mask__counts1/ensemble.parquet')
    rows.append(dict(modality='all_ehr',variant='E1_mask_only_LightGBM_3seed',**score(baseline.y_true,baseline.y_prob,baseline.threshold)))
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'comparisons.csv',index=False)
    report=['# Temporal mask control results',
      'Fixed 5 outer folds and 3 seeds averaged per patient; 6,225 patients and 705 deaths. AUPRC is average precision. Intervals are conditional 1,000-draw patient bootstrap intervals.',
      '| Modality | Input variant | AUROC | Average precision | Precision | Recall |',
      '|---|---|---:|---:|---:|---:|']
    for row in rows:
        report.append(f"| {row['modality']} | {row['variant']} | {row['auroc']:.4f} | {row['auprc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |")
    report += ['',
      'E1 mask-only uses both vitals and labs; it is a care-process reference, not a matched single-stream model. Its historical AUROC is 0.6867. Values-only removes explicit channels, while retaining the structural pooling mask.',
      'Entirely unobserved hours are masked in pooling, but imputed tokens remain visible to self-attention in all E2 arms. No architecture search was performed.',
      '## Paired differences',
      '| Modality and comparison | Delta AUROC with conditional CI | Fold deltas 0 through 4 | Positive negative zero folds | Interpretation |',
      '|---|---|---|---|---|']
    for row in comparisons:
        delta=' / '.join(f"{row[f'auroc_fold{fold}_delta']:+.4f}" for fold in range(5))
        signs='/'.join(str(row[f'auroc_{name}_folds']) for name in ('positive','negative','zero'))
        report.append(f"| {row['modality']} {row['comparison']} | {row['auroc_delta']:+.4f} [{row['auroc_lower']:+.4f}, {row['auroc_upper']:+.4f}] | {delta} | {signs} | {row['auroc_status']} |")
    report += ['', 'An unresolved difference is not evidence of equality. Pair-specific AUROC/AP resolution, all fold signs and intervals are in results/evaluation/mask_controls/comparisons.csv. Fold signs are descriptive because training sets overlap.']
    (P/'reports/MASK_CONTROLS.md').write_text('\n\n'.join(report).replace('|\n\n|','|\n|')+'\n')
    (out/'metadata.json').write_text(json.dumps({'historical_e1_mask_auroc':.686745,'warning':'E1 mask control uses both temporal modalities; not an identical input set to a single E2 stream',
       'pooling':'Entirely unobserved hours excluded from pooling in every E2 control. This preserves structural missingness in values-only.',
       'attention':'Imputed tokens remain visible to self-attention.'},indent=2))
    print('MASK_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__': main()
