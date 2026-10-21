"""Three independently fitted seeds with exactly the original E1 specification."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .baselines import load_data,design
from .cxr_baselines import fold_pca
from .tabular import lightgbm_fold
from .evaluate_e2 import average_seeds
from .metrics import select_threshold,score
from .fair_metrics import paired_evidence

P=Path('/projects/icumultimodal')
ARMS=[(('context',),False),(('vitals',),False),(('vitals',),True),
      (('labs',),False),(('labs',),True),(('cxr',),False),
      (('context','vitals','labs'),False),(('context','vitals','labs'),True),
      (('context','vitals','labs','cxr'),False),(('context','vitals','labs','cxr'),True),(('mask',),True)]


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('E1 ensembles require a CPU allocation')
    root=P/'results/e1_ensembles'
    root.mkdir(parents=True,exist_ok=True)
    data=load_data('main'); frame=data['frame']; y=frame.label.to_numpy()
    digest=hashlib.sha256(Path(__file__).read_bytes()+inspect.getsource(lightgbm_fold).encode()+inspect.getsource(design).encode()+(data['directory']/'partitions.npz').read_bytes())
    signature=digest.hexdigest(); lock=root/'signature.txt'
    if lock.exists() and lock.read_text().strip()!=signature:
        raise RuntimeError('E1 ensemble signature changed')
    lock.write_text(signature+'\n')
    pooled=np.load(P/'data/processed/cxr/pooled.npy',mmap_mode='r')
    image_indices=np.load(P/'data/processed/cxr/main_indices.npy',allow_pickle=False)
    embeddings=np.asarray(pooled[image_indices],dtype=np.float32)
    summaries=[]
    for arm,counts in ARMS:
        name='_'.join(arm)+f'__counts{int(counts)}'
        seed_oof=[]
        for seed in range(3):
            tests=[]
            for fold in range(5):
                directory=root/name/f'seed{seed}'/f'fold{fold}'
                directory.mkdir(parents=True,exist_ok=True)
                if not all((directory/file).exists() for file in ('test.parquet','validation.parquet','model.joblib')):
                    part={key:data['parts'][f'fold{fold}_{key}'] for key in ('train','validation','test')}
                    ehr=tuple(modality for modality in arm if modality!='cxr')
                    x,names,_=design(data,part,ehr,counts) if ehr else (np.empty((len(frame),0)),[],None)
                    if 'cxr' in arm:
                        reduced,_=fold_pca(P/'results/cxr_tabular',embeddings,part['train'],fold,'main')
                        x=np.concatenate([x,reduced],axis=1); names=names+[f'cxr_pc{i}' for i in range(64)]
                    fit=lightgbm_fold(x[part['train']],y[part['train']],x[part['validation']],y[part['validation']],x[part['test']],y[part['test']],names,include_missingness=counts,seed=seed)
                    joblib.dump(fit.estimator,directory/'model.joblib')
                    for portion,probability in [('test',fit.probabilities),('validation',fit.estimator.predict_proba(x[part['validation']])[:,1])]:
                        indices=part[portion]
                        result=frame.iloc[indices][['subject_id','stay_id','fold']].copy()
                        result['y_true'],result['y_prob']=y[indices],probability
                        for column,value in [('threshold',fit.threshold),('recall80_threshold',fit.recall80_threshold),('recall90_threshold',fit.recall90_threshold)]:
                            result[column]=value
                        temporary=directory/f'{portion}.part'
                        result.to_parquet(temporary,index=False); temporary.replace(directory/f'{portion}.parquet')
                tests.append(pd.read_parquet(directory/'test.parquet'))
            combined=pd.concat(tests).sort_values('subject_id')
            combined.to_parquet(root/name/f'seed{seed}.parquet',index=False)
            seed_oof.append(combined)
        ensemble=average_seeds(seed_oof)
        for fold in range(5):
            validation=average_seeds([pd.read_parquet(root/name/f'seed{seed}'/f'fold{fold}'/'validation.parquet') for seed in range(3)])
            for column,target in [('threshold',None),('recall80_threshold',.8),('recall90_threshold',.9)]:
                ensemble.loc[ensemble.fold==fold,column]=select_threshold(validation.y_true,validation.y_prob,target)
        ensemble.to_parquet(root/name/'ensemble.parquet',index=False)
        probabilities=np.array([rows.sort_values('subject_id').y_prob.to_numpy() for rows in seed_oof])
        record=dict(arm='_'.join(arm),include_counts=counts,name=name,seed_max_probability_difference=float(np.max(np.ptp(probabilities,axis=0))),**score(ensemble.y_true,ensemble.y_prob,ensemble.threshold))
        summaries.append(record)
        if record['auroc']>.98 or ('cxr' not in arm and arm!=('mask',) and record['auroc']>.90):
            raise RuntimeError('E1 ensemble exceeds a prespecified leakage gate')
        print('E1_ENSEMBLE_COMPLETE',name,flush=True)
    pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False)
    comparisons=[]
    for modality in ('context','vitals','labs','cxr'):
        first=pd.read_parquet(P/f'results/evaluation/e2/{modality}_ensemble.parquet')
        for arm,counts in ARMS:
            if arm!=(modality,): continue
            name='_'.join(arm)+f'__counts{int(counts)}'
            evidence=paired_evidence(first,pd.read_parquet(root/name/'ensemble.parquet'))
            comparisons.append(dict(comparison=f'e2_{modality}_minus_e1_lightgbm_{name}',**evidence))
    pd.DataFrame(comparisons).to_csv(root/'e2_paired_comparisons.csv',index=False)
    (root/'complete.json').write_text(json.dumps({'complete':True,'signature':signature,'seeds':[0,1,2],
      'note':'Original LightGBM has no feature or row subsampling. Seeds may yield identical models; effective ensemble diversity is reported, never assumed.'},indent=2))
    print('FAIR_E1_E2_COMPLETE',flush=True)


if __name__=='__main__':
    main()
