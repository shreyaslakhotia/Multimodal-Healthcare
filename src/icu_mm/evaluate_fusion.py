"""E3 aggregation and paired comparisons of locked predictions only."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
from .evaluate_e2 import average_seeds
from .aggregate import align_patients
from .bootstrap import patient_intervals,paired_bootstrap
from .metrics import select_threshold,score
from .fair_metrics import paired_evidence

P=Path('/projects/icumultimodal')
ARMS=('late_average','early','joint')


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Evaluation requires CPU Slurm')
    root=P/'results/deep/e3'
    completion=json.loads((root/'complete.json').read_text())
    if not completion['complete']:
        raise RuntimeError('E3 training incomplete')
    out=P/'results/evaluation/e3'
    out.mkdir(parents=True,exist_ok=True)
    figures=P/'reports/figures/e3'
    figures.mkdir(parents=True,exist_ok=True)
    records,folds,contrasts=[],[],[]
    predictions={}
    for arm in ARMS:
        cases=[]
        for seed in range(3):
            prediction=pd.read_parquet(P/f'results/oof/e3__{arm}__seed{seed}.parquet')
            if len(prediction)!=6225 or prediction.subject_id.duplicated().any():
                raise RuntimeError('E3 OOF support mismatch')
            cases.append(prediction)
            records.append(dict(arm=arm,estimator=f'seed{seed}',**score(prediction.y_true,prediction.y_prob,prediction.threshold)))
            for fold in range(5):
                metrics=json.loads((root/arm/f'fold{fold}'/f'seed{seed}'/'metrics.json').read_text())
                folds.append(dict(arm=arm,fold=fold,seed=seed,**metrics))
        ensemble=average_seeds(cases)
        for fold in range(5):
            validation=average_seeds([pd.read_parquet(root/arm/f'fold{fold}'/f'seed{seed}'/'validation.parquet') for seed in range(3)])
            for name,target in [('threshold',None),('recall80_threshold',.8),('recall90_threshold',.9)]:
                ensemble.loc[ensemble.fold==fold,name]=select_threshold(validation.y_true,validation.y_prob,target)
        ensemble.to_parquet(out/f'{arm}_ensemble.parquet',index=False)
        predictions[arm]=ensemble
        record=dict(arm=arm,estimator='mean_probability_ensemble',**score(ensemble.y_true,ensemble.y_prob,ensemble.threshold))
        interval=patient_intervals(ensemble.y_true,ensemble.subject_id,ensemble.y_prob)
        for metric in ('auroc','auprc'):
            for bound in ('lower','upper'):
                record[f'{metric}_{bound}']=interval[metric][bound]
        records.append(record)
        print('E3_EVALUATED',arm,flush=True)
    def compare(first,second,name):
        row={'comparison':name,**paired_evidence(first,second)}
        contrasts.append(row)
        print('E3_COMPARISON',name,flush=True)
    for first,second in [('early','late_average'),('joint','late_average'),('joint','early')]:
        compare(predictions[first],predictions[second],f'{first}_minus_{second}')
    fair_root=P/'results/e1_ensembles'
    if not json.loads((fair_root/'complete.json').read_text())['complete']:
        raise RuntimeError('Three-seed E1 comparators must be complete')
    for arm in ARMS:
        for baseline in ('context_vitals_labs','context_vitals_labs_cxr'):
            for counts in (0,1):
                name=f'{baseline}__counts{counts}'
                compare(predictions[arm],pd.read_parquet(fair_root/name/'ensemble.parquet'),f'{arm}_minus_e1_lightgbm_3seed_{name}')
    summary=pd.DataFrame(records)
    summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'comparisons.csv',index=False)
    report=['# E3 fixed fusion results',
      'Five patient-separated outer folds; three seeds averaged per patient; 6,225 patients and 705 deaths. All E3 arms include untimed CXR. No logistic stacking. AUPRC is average precision.',
      '| Arm | AUROC with conditional 95% CI | Average precision | Precision | Recall |',
      '|---|---|---:|---:|---:|']
    for row in records:
        if row['estimator']!='mean_probability_ensemble': continue
        report.append(f"| {row['arm']} | {row['auroc']:.4f} [{row['auroc_lower']:.4f}, {row['auroc_upper']:.4f}] | {row['auprc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |")
    report += ['', '## Paired comparisons',
       '| Comparison | Delta AUROC and conditional CI | Outer fold deltas 0 through 4 | Positive negative zero | Approximate resolution | Interpretation |',
       '|---|---|---|---|---|---|']
    for row in contrasts:
        delta=' / '.join(f"{row[f'auroc_fold{fold}_delta']:+.4f}" for fold in range(5))
        signs='/'.join(str(row[f'auroc_{name}_folds']) for name in ('positive','negative','zero'))
        report.append(f"| {row['comparison']} | {row['auroc_delta']:+.4f} [{row['auroc_lower']:+.4f}, {row['auroc_upper']:+.4f}] | {delta} | {signs} | {row['auroc_resolution_95_approx']:.4f} | {row['auroc_status']} |")
    report += ['', 'Intervals use 1,000 paired patient bootstrap draws conditional on fitted models. Fold signs are descriptive, not independent replications. A difference below pair-specific resolution or with a CI including zero is unresolved, never equal. No equivalence test was conducted. All paired AP results and per-seed/fold scores remain in the aggregate CSVs.',
      'The original deterministic E1 LightGBM produced identical predictions across its three seeds; its ensemble has no observed additional seed diversity. Null results are retained. No test score is used to redesign E4.',
      'Next primary RQ2 comparison is EHR-only E4a versus EHR-only jointly trained concatenation with matching E2 initializers. Full CXR E4 arms and E4b/E5 remain secondary/sensitivity analyses as declared in docs/E4_DESIGN.md.']
    (P/'reports/E3_RESULTS.md').write_text('\n\n'.join(report).replace('|\n\n|','|\n|')+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for arm in ARMS:
        data=predictions[arm]
        row=summary.loc[(summary.arm==arm)&(summary.estimator=='mean_probability_ensemble')].iloc[0]
        x,y,_=roc_curve(data.y_true,data.y_prob)
        axes[0].plot(x,y,label=f'{arm} ({row.auroc:.3f})')
        precision,recall,_=precision_recall_curve(data.y_true,data.y_prob)
        axes[1].plot(recall,precision,label=f'{arm} ({row.auprc:.3f})')
    axes[0].plot([0,1],[0,1],'--',color='gray')
    axes[1].axhline(705/6225,ls='--',color='gray')
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='E3 ROC and AUROC')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='E3 precision recall and average precision')
    for ax in axes:
        ax.legend(); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    fig.text(.5,.01,'6,225 patients, 705 deaths; 5 patient folds; 3 seeds averaged per patient. All arms include untimed CXR.\nLocked OOF predictions; no confidence bands; paired patient intervals are in comparisons.csv.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1))
    fig.savefig(figures/'roc_pr.png',dpi=170,bbox_inches='tight'); plt.close(fig)
    (out/'metadata.json').write_text(json.dumps({'training':completion,'patients':6225,'deaths':705,'draws':1000,'multiple_comparisons':'descriptive, not multiplicity adjusted'},indent=2))
    print('E3_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
