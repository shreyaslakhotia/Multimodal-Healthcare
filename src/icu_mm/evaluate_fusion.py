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
        paired=align_patients(first,second)
        result=paired_bootstrap(paired.y_true_first,paired.subject_id,paired.y_prob_first,paired.y_prob_second)
        row={'comparison':name}
        for metric in ('auroc','auprc'):
            for key in ('delta','lower','upper'):
                row[f'{metric}_{key}']=result[metric][key]
        contrasts.append(row)
        print('E3_COMPARISON',name,flush=True)
    for first,second in [('early','late_average'),('joint','late_average'),('joint','early')]:
        compare(predictions[first],predictions[second],f'{first}_minus_{second}')
    table=pd.read_csv(P/'results/summary_table.csv')
    baselines=table.loc[table.arm.isin(['context_vitals_labs','context_vitals_labs_cxr'])&(table.selection=='main')]
    for arm in ('early','joint'):
        for _,row in baselines.iterrows():
            compare(predictions[arm],pd.read_parquet(P/f'results/oof/{row.experiment}__seed0.parquet'),f'{arm}_minus_{row.experiment}')
    summary=pd.DataFrame(records)
    summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'comparisons.csv',index=False)
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
