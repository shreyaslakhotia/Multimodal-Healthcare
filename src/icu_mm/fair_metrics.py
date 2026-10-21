"""Conditional paired intervals, outer-fold signs and approximate resolution."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .aggregate import align_patients
from .bootstrap import paired_bootstrap


def paired_evidence(first,second,iterations=1000):
    paired=align_patients(first,second)
    interval=paired_bootstrap(paired.y_true_first,paired.subject_id,paired.y_prob_first,paired.y_prob_second,iterations=iterations)
    record={}
    for name,metric in [('auroc',roc_auc_score),('auprc',average_precision_score)]:
        differences=[]
        for fold in range(5):
            rows=paired.loc[paired.fold_first==fold]
            if not len(rows) or rows.y_true_first.nunique()!=2:
                raise ValueError('Every outer fold must contain both classes')
            value=float(metric(rows.y_true_first,rows.y_prob_first)-metric(rows.y_true_first,rows.y_prob_second))
            record[f'{name}_fold{fold}_delta']=value
            differences.append(value)
        values=interval[name]
        for key in ('delta','lower','upper'):
            record[f'{name}_{key}']=values[key]
        half=max(values['delta']-values['lower'],values['upper']-values['delta'])
        record[f'{name}_resolution_95_approx']=half
        record[f'{name}_resolution_80power_normal_approx']=half*(1.96+.84)/1.96
        record[f'{name}_positive_folds']=int((np.asarray(differences)>0).sum())
        record[f'{name}_negative_folds']=int((np.asarray(differences)<0).sum())
        record[f'{name}_zero_folds']=int((np.asarray(differences)==0).sum())
        record[f'{name}_status']='unresolved' if values['lower']<=0<=values['upper'] or abs(values['delta'])<half else 'conditional_interval_excludes_zero'
    return record
