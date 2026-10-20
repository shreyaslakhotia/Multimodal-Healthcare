"""CXR-only, selection sensitivity and complete tabular modality arms."""
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .baselines import P,LEARNERS,load_data,design
from .guards import assert_feature_names
from .metrics import score


def fold_pca(cache,embeddings,train,fold,selection):
    train_hash=hashlib.sha256(np.asarray(train,dtype=np.int64).tobytes()).hexdigest()
    model_path=cache/f'pca_{selection}_fold{fold}.joblib'
    values_path=cache/f'pca_{selection}_fold{fold}.npy'
    if model_path.exists() and values_path.exists():
        pca=joblib.load(model_path)
        if getattr(pca,'training_index_hash_',None)!=train_hash:
            raise RuntimeError('Cached CXR PCA was fitted on different training rows')
        return np.load(values_path,mmap_mode='r'),pca
    pca=PCA(n_components=64,svd_solver='randomized',random_state=0)
    pca.fit(embeddings[train])
    pca.training_index_hash_=train_hash
    transformed=pca.transform(embeddings).astype(np.float32)
    joblib.dump(pca,model_path)
    np.save(values_path,transformed,allow_pickle=False)
    return transformed,pca


def run_arm(data,embeddings,arm,learner,counts,selection,root):
    name='_'.join(arm)
    experiment=f'main__{name}__{learner}__counts{int(counts)}__selection{selection}'
    directory=root/experiment
    directory.mkdir(exist_ok=True)
    frame=data['frame']
    predictions,fold_scores=[],[]
    for fold in range(5):
        prediction_path=directory/f'fold{fold}.parquet'
        metric_path=directory/f'fold{fold}.json'
        model_path=directory/f'fold{fold}.joblib'
        if all(path.exists() for path in (prediction_path,metric_path,model_path)):
            predictions.append(pd.read_parquet(prediction_path))
            fold_scores.append(json.loads(metric_path.read_text()))
            continue
        part={key:data['parts'][f'fold{fold}_{key}'] for key in ('train','validation','test')}
        groups=[set(frame.iloc[part[key]].subject_id) for key in part]
        if any(groups[a]&groups[b] for a,b in [(0,1),(0,2),(1,2)]):
            raise RuntimeError('Patient overlap in CXR experiment partitions')
        reduced,pca=fold_pca(root,embeddings,part['train'],fold,selection)
        base_arm=tuple(modality for modality in arm if modality!='cxr')
        if base_arm:
            base,names,context=design(data,part,base_arm,counts)
        else:
            base,names,context=np.empty((len(frame),0)),[],None
        x=np.concatenate([base,reduced],axis=1)
        names=names+[f'cxr_pc{i}' for i in range(64)]
        assert_feature_names(names)
        y=frame.label.to_numpy()
        fit=LEARNERS[learner](x[part['train']],y[part['train']],x[part['validation']],y[part['validation']],
                              x[part['test']],y[part['test']],names,include_missingness=counts,seed=0)
        prediction=frame.iloc[part['test']][['stay_id','subject_id','fold']].copy()
        prediction['y_true']=y[part['test']]
        prediction['y_prob']=fit.probabilities
        prediction['threshold']=fit.threshold
        prediction['recall80_threshold']=fit.recall80_threshold
        prediction['recall90_threshold']=fit.recall90_threshold
        joblib.dump({'estimator':fit.estimator,'context':context,'pca':pca,'feature_names':names},model_path)
        prediction.to_parquet(prediction_path,index=False)
        metric_path.write_text(json.dumps(fit.metrics,indent=2))
        predictions.append(prediction)
        fold_scores.append(fit.metrics)
        print('CXR_FOLD_COMPLETE',experiment,fold,flush=True)
    oof=pd.concat(predictions).sort_values('subject_id').reset_index(drop=True)
    if len(oof)!=len(frame) or oof.subject_id.duplicated().any():
        raise RuntimeError('Invalid CXR OOF patient coverage')
    oof.to_parquet(P/f'results/oof/{experiment}__seed0.parquet',index=False)
    record={'experiment':experiment,'cohort':'main','arm':name,'model':learner,
            'include_counts':counts,'selection':selection,'shuffled_training_labels':False,
            **score(oof.y_true,oof.y_prob,oof.threshold.to_numpy())}
    for operating in ('recall80','recall90'):
        metrics=score(oof.y_true,oof.y_prob,oof[f'{operating}_threshold'].to_numpy())
        record[f'{operating}_precision']=metrics['precision']
        record[f'{operating}_recall']=metrics['recall']
    for metric in ('auroc','auprc','precision','recall','f1','specificity'):
        record[f'{metric}_fold_mean']=float(np.mean([row[metric] for row in fold_scores]))
        record[f'{metric}_fold_std']=float(np.std([row[metric] for row in fold_scores],ddof=1))
    (directory/'summary.json').write_text(json.dumps(record,indent=2))
    if record['auroc']>.98:
        raise RuntimeError('Near-perfect CXR model score requires leakage investigation')
    return record


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('CXR tabular fitting requires a CPU allocation')
    cache=P/'data/processed/cxr'
    manifest=json.loads((cache/'manifest.json').read_text())
    if not manifest['complete'] or manifest['weights']!='densenet121-res224-chex':
        raise RuntimeError('Verified non-MIMIC frozen CXR cache required')
    data=load_data('main')
    root=P/'results/cxr_tabular'
    root.mkdir(exist_ok=True)
    digest=hashlib.sha256()
    for path in (Path(__file__),Path('src/icu_mm/tabular.py'),Path('src/icu_mm/baselines.py'),Path('src/icu_mm/context.py'),Path('src/icu_mm/metrics.py'),data['directory']/'partitions.npz'):
        digest.update(path.read_bytes())
    digest.update((manifest['signature']+manifest['weights_sha256']).encode())
    signature=digest.hexdigest()
    lock=root/'signature.txt'
    if lock.exists() and lock.read_text().strip()!=signature:
        raise RuntimeError('CXR experiment implementation changed after locking')
    lock.write_text(signature+'\n')
    pooled=np.load(cache/'pooled.npy',mmap_mode='r')
    rows=[]
    for selection in ('main','alternate'):
        indices=np.load(cache/f'{selection}_indices.npy',allow_pickle=False)
        if len(indices)!=len(data['frame']):
            raise RuntimeError('CXR index maps disagree with cohort length')
        embeddings=np.asarray(pooled[indices],dtype=np.float32)
        arms=[('cxr',)] if selection=='alternate' else [('cxr',),('context','cxr'),('vitals','labs','cxr'),('context','vitals','cxr'),('context','labs','cxr'),('context','vitals','labs','cxr')]
        for learner in LEARNERS:
            for arm in arms:
                variants=(False,True) if 'vitals' in arm or 'labs' in arm else (False,)
                for counts in variants:
                    rows.append(run_arm(data,embeddings,arm,learner,counts,selection,root))
    pd.DataFrame(rows).to_csv(P/'results/cxr_tabular_initial_summary.csv',index=False)
    print('CXR_CPU_BASELINES_COMPLETE',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        reason=str(error) if isinstance(error,RuntimeError) else type(error).__name__
        (P/'reports/BLOCKED.md').write_text(f'CXR baselines stopped: {reason}. No automatic retries.\n')
        print('CXR_BASELINES_STOPPED',reason,flush=True)
        raise SystemExit(1)
