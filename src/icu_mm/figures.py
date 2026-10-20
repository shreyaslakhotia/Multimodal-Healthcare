"""Regenerable aggregate research figures from locked OOF predictions."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve

P=Path('/projects/icumultimodal')
LABELS={'context':'Context','vitals':'Vitals','labs':'Labs','cxr':'CXR (untimed)',
        'context_vitals_labs':'EHR','context_vitals_labs_cxr':'All (untimed CXR)'}
COLORS=['#6b7280','#2563eb','#059669','#d97706','#7c3aed','#dc2626']


def oof(row):
    return pd.read_parquet(P/f'results/oof/{row.experiment}__seed0.parquet')


def save(fig,name):
    destination=P/'reports/figures'
    destination.mkdir(exist_ok=True)
    fig.savefig(destination/f'{name}.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Figure generation requires a CPU allocation')
    table=pd.read_csv(P/'results/summary_table.csv')
    table=table.loc[table.selection=='main']
    contrast=pd.read_csv(P/'results/modality_contrasts.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for counts in (False,True):
        fig,axes=plt.subplots(2,2,figsize=(12,10))
        for row_index,learner in enumerate(('logistic','lightgbm')):
            for (arm,label),color in zip(LABELS.items(),COLORS):
                wanted_counts=False if arm in ('context','cxr') else counts
                row=table.loc[(table.model==learner)&(table.arm==arm)&(table.include_counts==wanted_counts)].iloc[0]
                values=oof(row)
                fpr,tpr,_=roc_curve(values.y_true,values.y_prob)
                precision,recall,_=precision_recall_curve(values.y_true,values.y_prob)
                axes[row_index,0].plot(fpr,tpr,label=f'{label}: {row.auroc:.3f}',color=color,lw=1.7)
                axes[row_index,1].plot(recall,precision,label=f'{label}: {row.auprc:.3f}',color=color,lw=1.7)
            axes[row_index,0].plot([0,1],[0,1],ls='--',color='#9ca3af',lw=.8)
            axes[row_index,1].axhline(values.y_true.mean(),ls='--',color='#9ca3af',lw=.8)
            for column in range(2):
                ax=axes[row_index,column]
                ax.set(xlim=(0,1),ylim=(0,1),xlabel='False positive rate' if column==0 else 'Recall',
                       ylabel='True positive rate' if column==0 else 'Precision',title=learner.title()+(' — ROC' if column==0 else ' — PR'))
                ax.legend(fontsize=8,loc='lower right' if column==0 else 'upper right')
                ax.grid(alpha=.15)
        fig.suptitle('Tabular modality arms — '+('with' if counts else 'without')+' explicit counts/indicators')
        fig.text(.5,.015,'6,225 patients; five patient-separated folds; seed 0; pooled OOF curves.\nNo interval bands; 95% patient-bootstrap intervals are in the summary table. CXR is untimed.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.055,1,.96))
        save(fig,f'tabular_roc_pr_counts{int(counts)}')
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for row_index,learner in enumerate(('logistic','lightgbm')):
        for column,counts in enumerate((False,True)):
            row=table.loc[(table.model==learner)&(table.arm=='context_vitals_labs_cxr')&(table.include_counts==counts)].iloc[0]
            matrix=np.array([[row.tn,row.fp],[row.fn,row.tp]],dtype=int)
            ax=axes[row_index,column]
            ax.imshow(matrix,cmap='Blues',vmin=0)
            for (i,j),value in np.ndenumerate(matrix):
                ax.text(j,i,f'{value:,}',ha='center',va='center',fontsize=13,color='white' if value>matrix.max()/2 else 'black')
            ax.set(xticks=[0,1],xticklabels=['Survival','Death'],yticks=[0,1],yticklabels=['Survival','Death'],
                   xlabel='Predicted',ylabel='Observed',title=f'{learner.title()}, counts={counts}')
    fig.suptitle('All modalities: pooled confusion counts — untimed CXR')
    fig.text(.5,.01,'6,225 patients; five folds; seed 0. Each fold uses its validation maximum-F1 threshold.\nCounts have no interval bars; no test threshold selection.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.95))
    save(fig,'tabular_confusion_matrices')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    keys=['full_context_leave_out','full_vitals_leave_out','full_labs_leave_out','full_cxr_leave_out']
    for row_index,learner in enumerate(('logistic','lightgbm')):
        for column,counts in enumerate((False,True)):
            values=contrast.loc[(contrast.model==learner)&(contrast.include_counts==counts)].set_index('comparison').loc[keys]
            estimate=values.auroc_delta.to_numpy()
            ax=axes[row_index,column]
            ax.barh(np.arange(4),estimate,color=COLORS[:4],alpha=.8)
            ax.errorbar(estimate,np.arange(4),xerr=np.vstack([estimate-values.auroc_lower.to_numpy(),values.auroc_upper.to_numpy()-estimate]),fmt='none',ecolor='#111827',capsize=4)
            ax.axvline(0,color='#6b7280',ls='--',lw=1)
            ax.set(yticks=np.arange(4),yticklabels=['Context','Vitals','Labs','CXR (untimed)'],
                   xlabel='Full minus leave-one-out AUROC',title=f'{learner.title()}, counts={counts}')
            ax.invert_yaxis()
            ax.grid(axis='x',alpha=.15)
    fig.suptitle('Tabular modality contributions — association, not causal effects')
    fig.text(.5,.01,'6,225 patients; identical five folds; seed 0. Error bars: 95% paired patient-bootstrap CIs, 1,000 draws.\nEach ablation is retrained with fixed hyperparameters. CXR timestamps are unavailable.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95))
    save(fig,'tabular_modality_importance')
    print('TABULAR_FIGURES_READY',flush=True)


if __name__=='__main__':
    main()
