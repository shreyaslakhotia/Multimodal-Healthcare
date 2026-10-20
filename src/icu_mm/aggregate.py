"""Aggregate locked predictions; never refit or select a model on test data."""
import json
import os
from pathlib import Path
import pandas as pd
from .bootstrap import paired_bootstrap,patient_intervals

P=Path('/projects/icumultimodal')


def align_patients(first,second):
    paired=first.merge(second,on='subject_id',suffixes=('_first','_second'),validate='one_to_one')
    if len(paired)!=len(first) or len(first)!=len(second):
        raise ValueError('Patient sets differ between paired models')
    if not (paired.y_true_first==paired.y_true_second).all() or not (paired.fold_first==paired.fold_second).all():
        raise ValueError('Labels or folds differ between paired models')
    return paired


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Aggregation requires a CPU Slurm allocation')
    initial=pd.read_csv(P/'results/tabular_initial_summary.csv')
    selected=initial.loc[(initial.cohort=='main') & ~initial.shuffled_training_labels & (initial.arm!='mask')].copy()
    cache=P/'results/bootstrap'
    cache.mkdir(exist_ok=True)
    for index,row in selected.iterrows():
        filename=cache/f'{row.experiment}.json'
        if filename.exists():
            interval=json.loads(filename.read_text())
        else:
            oof=pd.read_parquet(P/f'results/oof/{row.experiment}__seed0.parquet')
            interval=patient_intervals(oof.y_true,oof.subject_id,oof.y_prob)
            filename.write_text(json.dumps(interval,indent=2))
        for metric in ('auroc','auprc'):
            for bound in ('lower','upper'):
                selected.loc[index,f'{metric}_ci_{bound}']=interval[metric][bound]
        selected.loc[index,'bootstrap_valid_draws']=interval['valid_draws']
        print('BOOTSTRAP_COMPLETE',row.experiment,flush=True)
    selected.to_csv(P/'results/summary_table.csv',index=False)
    contrasts=[('context_vitals_labs','vitals_labs','context_leave_out'),
               ('context_vitals_labs','context_labs','vitals_leave_out'),
               ('context_vitals_labs','context_vitals','labs_leave_out'),
               ('context_vitals','context','vitals_add_to_context'),
               ('context_labs','context','labs_add_to_context'),
               ('context_vitals_labs','context','vitals_labs_add_to_context')]
    comparisons=[]
    for learner in ('logistic','lightgbm'):
        for counts in (False,True):
            for first_arm,second_arm,name in contrasts:
                first=selected.loc[(selected.model==learner)&(selected.arm==first_arm)&(selected.include_counts==counts)].iloc[0]
                second_counts=False if second_arm=='context' else counts
                second=selected.loc[(selected.model==learner)&(selected.arm==second_arm)&(selected.include_counts==second_counts)].iloc[0]
                comparison_path=cache/f'paired_{learner}_{int(counts)}_{name}.json'
                if comparison_path.exists():
                    result=json.loads(comparison_path.read_text())
                else:
                    paired=align_patients(pd.read_parquet(P/f'results/oof/{first.experiment}__seed0.parquet'),
                                          pd.read_parquet(P/f'results/oof/{second.experiment}__seed0.parquet'))
                    result=paired_bootstrap(paired.y_true_first,paired.subject_id,paired.y_prob_first,paired.y_prob_second)
                    comparison_path.write_text(json.dumps(result,indent=2))
                record={'model':learner,'include_counts':counts,'comparison':name,
                        'first':first.experiment,'second':second.experiment}
                for metric in ('auroc','auprc'):
                    for key in ('delta','lower','upper'):
                        record[f'{metric}_{key}']=result[metric][key]
                comparisons.append(record)
    pd.DataFrame(comparisons).to_csv(P/'results/modality_contrasts.csv',index=False)
    print('INITIAL_EHR_AGGREGATION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
