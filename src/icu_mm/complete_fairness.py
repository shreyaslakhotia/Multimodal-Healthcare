"""Add fold evidence for prior deterministic logistic reference claims."""
import os
from pathlib import Path
import pandas as pd
from .fair_metrics import paired_evidence

P=Path('/projects/icumultimodal')


def main():
    if 'SLURM_JOB_ID' not in os.environ: raise RuntimeError('CPU allocation required')
    records=[]
    table=pd.read_csv(P/'results/summary_table.csv')
    for modality in ('context','vitals','labs','cxr'):
        model=pd.read_parquet(P/f'results/evaluation/e2/{modality}_ensemble.parquet')
        for _,row in table.loc[(table.model=='logistic')&(table.arm==modality)&(table.selection=='main')].iterrows():
            baseline=pd.read_parquet(P/f'results/oof/{row.experiment}__seed0.parquet')
            records.append(dict(comparison=f'e2_{modality}_minus_{row.experiment}',**paired_evidence(model,baseline)))
    pd.DataFrame(records).to_csv(P/'results/e1_ensembles/e2_logistic_fold_comparisons.csv',index=False)
    print('LEGACY_LOGISTIC_FOLD_EVIDENCE_COMPLETE',flush=True)


if __name__=='__main__': main()
