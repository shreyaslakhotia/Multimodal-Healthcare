"""CPU canaries and fixed EHR tabular arms on persisted patient partitions."""
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .context import ContextEncoder
from .guards import assert_feature_names
from .metrics import score
from .tabular import logistic_fold,lightgbm_fold

P=Path('/projects/icumultimodal')
LEARNERS={'logistic':logistic_fold,'lightgbm':lightgbm_fold}


def load_data(name):
    directory=P/'data/processed'/name
    frame=pd.read_parquet(directory/'cohort.parquet')
    if frame.subject_id.duplicated().any():
        raise RuntimeError('Repeated patients in persisted cohort')
    return {'frame':frame,'directory':directory,'parts':np.load(directory/'partitions.npz',allow_pickle=False)}


def design(data,part,arm,include_counts):
    frame=data['frame']
    context=ContextEncoder().fit(frame.iloc[part['train']])
    suffix='withcounts' if include_counts else 'withoutcounts'
    definition=json.loads((data['directory']/f'summary_{suffix}.json').read_text())
    values=np.load(data['directory']/f'summary_{suffix}.npy',mmap_mode='r')
    matrices,names=[],[]
    if 'context' in arm:
        matrices.append(context.transform(frame))
        names+=context.feature_names_
    if 'vitals' in arm:
        matrices.append(values[:,:len(definition['vitals'])])
        names+=definition['vitals']
    if 'labs' in arm:
        matrices.append(values[:,len(definition['vitals']):])
        names+=definition['labs']
    if arm==('mask',):
        for modality in ('vitals','labs'):
            mask=np.load(data['directory']/f'{modality}_mask.npy',mmap_mode='r')
            matrices.extend([mask.reshape(len(frame),-1),mask.sum(axis=1)])
            names+=[f'{modality}_mask_h{hour}_c{channel}' for hour in range(48) for channel in range(mask.shape[2])]
            names+=[f'{modality}_count_c{channel}' for channel in range(mask.shape[2])]
    if not matrices:
        raise RuntimeError('No model inputs selected')
    assert_feature_names(names)
    return np.concatenate(matrices,axis=1),names,context


def run_arm(data,cohort_name,arm,learner,counts,shuffle=False):
    arm_name='_'.join(arm)
    experiment=f'{cohort_name}__{arm_name}__{learner}__counts{int(counts)}'+('__shuffled' if shuffle else '')
    directory=P/'results/tabular'/experiment
    directory.mkdir(parents=True,exist_ok=True)
    frame=data['frame']
    predictions,fold_metrics=[],[]
    for fold in range(5):
        prediction_path=directory/f'fold{fold}.parquet'
        metric_path=directory/f'fold{fold}.json'
        model_path=directory/f'fold{fold}.joblib'
        if prediction_path.exists() and metric_path.exists() and model_path.exists():
            predictions.append(pd.read_parquet(prediction_path))
            fold_metrics.append(json.loads(metric_path.read_text()))
            continue
        part={name:data['parts'][f'fold{fold}_{name}'] for name in ('train','validation','test')}
        if any(len(np.unique(indices))!=len(indices) for indices in part.values()):
            raise RuntimeError('Repeated row indices inside a persisted partition')
        sets=[set(frame.iloc[part[name]].subject_id) for name in part]
        if any(sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)]):
            raise RuntimeError('Patient overlap in persisted partitions')
        if set(np.concatenate(list(part.values())))!=set(range(len(frame))):
            raise RuntimeError('Partitions do not cover exactly the cohort')
        if not (frame.iloc[part['test']].fold==fold).all():
            raise RuntimeError('Test indices disagree with stored fold IDs')
        x,names,context=design(data,part,arm,counts)
        y=frame.label.to_numpy()
        training_y=y[part['train']].copy()
        if shuffle:
            np.random.default_rng(42+fold).shuffle(training_y)
        fit=LEARNERS[learner](x[part['train']],training_y,x[part['validation']],y[part['validation']],
                              x[part['test']],y[part['test']],names,include_missingness=counts,seed=0)
        prediction=frame.iloc[part['test']][['stay_id','subject_id','fold']].copy()
        prediction['y_true']=y[part['test']]
        prediction['y_prob']=fit.probabilities
        prediction['threshold']=fit.threshold
        prediction['recall80_threshold']=fit.recall80_threshold
        prediction['recall90_threshold']=fit.recall90_threshold
        # Persist the fitted fold before marking its result complete.
        joblib.dump({'estimator':fit.estimator,'context':context,'feature_names':names},model_path)
        prediction.to_parquet(prediction_path,index=False)
        metric_path.write_text(json.dumps(fit.metrics,indent=2))
        predictions.append(prediction)
        fold_metrics.append(fit.metrics)
        print('FOLD_COMPLETE',experiment,fold,flush=True)
    oof=pd.concat(predictions).sort_values('subject_id').reset_index(drop=True)
    if oof.subject_id.duplicated().any() or len(oof)!=len(frame):
        raise RuntimeError('OOF predictions are not one per patient')
    oof_dir=P/'results/oof'
    oof_dir.mkdir(exist_ok=True)
    oof.to_parquet(oof_dir/f'{experiment}__seed0.parquet',index=False)
    pooled=score(oof.y_true,oof.y_prob,oof.threshold.to_numpy())
    record={'experiment':experiment,'cohort':cohort_name,'arm':arm_name,'model':learner,
            'include_counts':counts,'shuffled_training_labels':shuffle,**pooled}
    for operating in ('recall80','recall90'):
        operating_score=score(oof.y_true,oof.y_prob,oof[f'{operating}_threshold'].to_numpy())
        record[f'{operating}_precision']=operating_score['precision']
        record[f'{operating}_recall']=operating_score['recall']
    for metric in ('auroc','auprc','precision','recall','f1','specificity'):
        record[f'{metric}_fold_mean']=float(np.mean([row[metric] for row in fold_metrics]))
        record[f'{metric}_fold_std']=float(np.std([row[metric] for row in fold_metrics],ddof=1))
    (directory/'summary.json').write_text(json.dumps(record,indent=2))
    if cohort_name=='main' and not shuffle and pooled['auroc']>.98:
        raise RuntimeError('Unexplained near-perfect main-cohort AUROC')
    return record


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Baseline fitting requires a CPU Slurm allocation')
    output=P/'results/tabular'
    output.mkdir(parents=True,exist_ok=True)
    feature_report=json.loads((P/'results/cohort/features.json').read_text())
    # Lock both fixed model implementation and actual partition bytes.
    digest=hashlib.sha256()
    for path in [Path('src/icu_mm/tabular.py'),Path('src/icu_mm/baselines.py'),Path('src/icu_mm/context.py'),
                 Path('src/icu_mm/metrics.py'),Path('requirements.txt'),
                 P/'data/processed/main/partitions.npz',P/'data/processed/s0/partitions.npz']:
        digest.update(path.read_bytes())
    digest.update(feature_report['config_hash'].encode())
    signature=digest.hexdigest()
    lock=output/'signature.txt'
    if lock.exists() and lock.read_text().strip()!=signature:
        raise RuntimeError('Experiment implementation or partitions changed after results were locked')
    lock.write_text(signature+'\n')
    main_data,s0=load_data('main'),load_data('s0')
    records=[]
    for learner in LEARNERS:
        record=run_arm(main_data,'main',('context','vitals','labs'),learner,True,shuffle=True)
        records.append(record)
        if not .45<=record['auroc']<=.55:
            raise RuntimeError('Shuffled-label canary outside prespecified AUROC band')
    for learner in LEARNERS:
        records.append(run_arm(s0,'s0',('mask',),learner,True))
        records.append(run_arm(main_data,'main',('mask',),learner,True))
    # EHR canary gates the remainder; context and EHR outputs are also E1 arms.
    for learner in LEARNERS:
        for counts in (False,True):
            record=run_arm(main_data,'main',('context','vitals','labs'),learner,counts)
            records.append(record)
            if record['auroc']>.90:
                raise RuntimeError('EHR-only AUROC exceeds prespecified leakage stop threshold')
    for learner in LEARNERS:
        records.append(run_arm(main_data,'main',('context',),learner,False))
        for arm in [('vitals',),('labs',),('vitals','labs'),('context','vitals'),('context','labs')]:
            for counts in (False,True):
                records.append(run_arm(main_data,'main',arm,learner,counts))
    pd.DataFrame(records).to_csv(P/'results/tabular_initial_summary.csv',index=False)
    canaries={'shuffled':[{key:row[key] for key in ('model','auroc','auprc')} for row in records if row['shuffled_training_labels']],
              'mask_only':[{key:row[key] for key in ('cohort','model','auroc','auprc')} for row in records if row['arm']=='mask'],
              'final_mask_warning':any(row['cohort']=='main' and row['arm']=='mask' and row['auroc']>.65 for row in records),
              'status':'INITIAL_CPU_ARMS_COMPLETE_CXR_AND_BOOTSTRAP_PENDING'}
    (P/'results/canaries.json').write_text(json.dumps(canaries,indent=2))
    print('CPU_BASELINES_COMPLETE',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        reason=str(error) if isinstance(error,RuntimeError) else type(error).__name__
        (P/'reports/BLOCKED.md').write_text(f'CPU baselines stopped: {reason}. No automatic retries.\n')
        print('BASELINES_STOPPED',reason,flush=True)
        raise SystemExit(1)
