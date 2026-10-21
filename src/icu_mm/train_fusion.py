"""Fixed E3 early and joint fusion, initialized from locked E2 encoders."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from .fusion import ConcatenationFusion
from .models import MODALITIES
from .train_unimodal import prepare_fold, make_loader, seed_all, rng_state, restore_rng, atomic_save
from .metrics import score, select_threshold
from .aggregate import align_patients

P=Path('/projects/icumultimodal')
CONFIG={'arms':['early','joint'],'seeds':[0,1,2],'folds':5,'batch_size':64,'max_epochs':50,
        'patience':10,'learning_rate':1e-4,'weight_decay':.01,'clip':1.,'precision':'bf16',
        'head':[512,128,1],'dropout':.1,'initialization':'same fold and seed E2 encoders',
        'backbone':'CheXpert DenseNet remains frozen; joint updates E2 encoders only'}


class FusionInputs(Dataset):
    def __init__(self, indices, features, valid, labels, cxr_indices, embedding=None):
        self.indices=np.asarray(indices)
        self.features,self.valid,self.labels=features,valid,labels
        self.cxr_indices,self.embedding=cxr_indices,embedding
        self.tokens=None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        row=self.indices[index]
        if self.embedding is not None:
            values={'embedding':torch.tensor(self.embedding[row],dtype=torch.float32)}
            masks={}
        else:
            if self.tokens is None:
                self.tokens=np.load(P/'data/processed/cxr/tokens.npy',mmap_mode='r')
            values={name:torch.tensor(self.features[name][row],dtype=torch.float32) for name in ('context','vitals','labs')}
            values['cxr']=torch.tensor(np.array(self.tokens[self.cxr_indices[row]],copy=True),dtype=torch.float32)
            masks={name:torch.as_tensor(self.valid[name][row],dtype=torch.bool) for name in ('context','vitals','labs')}
            masks['cxr']=torch.ones(49,dtype=torch.bool)
        return values,masks,torch.tensor(self.labels[row],dtype=torch.float32)


def device_inputs(values,valid,device):
    return ({name:value.to(device,non_blocking=True) for name,value in values.items()},
            {name:value.to(device,non_blocking=True) for name,value in valid.items()})


def predict(model,loader,device):
    model.eval()
    probabilities=[]
    with torch.inference_mode():
        for values,valid,_ in loader:
            values,valid=device_inputs(values,valid,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(values,valid)
            probabilities.append(torch.sigmoid(output.float()).cpu().numpy())
    return np.concatenate(probabilities)


def load_encoders(model,fold,seed):
    root=P/'results/deep/e2'
    for modality in MODALITIES:
        weights=torch.load(root/modality/f'fold{fold}'/f'seed{seed}'/'encoder.pt',map_location='cpu',weights_only=True)
        getattr(model.encoders,modality).load_state_dict(weights,strict=True)


def cached_embeddings(model,fold,seed,features,valid,y,cxr_indices,device,root):
    path=root/f'embeddings_fold{fold}_seed{seed}.npy'
    if path.exists():
        values=np.load(path,allow_pickle=False)
        if values.shape!=(len(y),512) or not np.isfinite(values).all():
            raise RuntimeError('Frozen embedding cache invalid')
        return values
    model.eval()
    loader=make_loader(FusionInputs(np.arange(len(y)),features,valid,y,cxr_indices),False)
    chunks=[]
    with torch.inference_mode():
        for values,mask,_ in loader:
            values,mask=device_inputs(values,mask,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                embedded=model.encode(values,mask)
            chunks.append(embedded.float().cpu().numpy())
    result=np.concatenate(chunks)
    temporary=path.with_suffix('.part')
    with temporary.open('wb') as handle:
        np.save(handle,result,allow_pickle=False)
    temporary.replace(path)
    return result


def fit_case(root,frame,part,features,valid,cxr_indices,arm,fold,seed):
    directory=root/arm/f'fold{fold}'/f'seed{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    if all((directory/name).exists() for name in ('test.parquet','validation.parquet','metrics.json','best.pt')):
        metrics=json.loads((directory/'metrics.json').read_text())
        if metrics['auroc']>.98:
            raise RuntimeError('Persisted E3 AUROC exceeds stop threshold')
        return
    seed_all(seed)
    device=torch.device('cuda')
    model=ConcatenationFusion(features['context'].shape[1],frozen=arm=='early').to(device)
    load_encoders(model,fold,seed)
    y=frame.label.to_numpy()
    embeddings=cached_embeddings(model,fold,seed,features,valid,y,cxr_indices,device,root) if arm=='early' else None
    generator=torch.Generator().manual_seed(42+seed+100*fold)
    def dataset(indices):
        return FusionInputs(indices,features if embeddings is None else None,valid if embeddings is None else None,y,cxr_indices,embeddings)
    training=make_loader(dataset(part['train']),True,generator)
    validation=make_loader(dataset(part['validation']),False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    positives=y[part['train']].sum()
    if positives==0 or positives==len(part['train']):
        raise RuntimeError('Both training classes required')
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(part['train'])-positives)/positives,device=device))
    latest=directory/'latest.pt'
    best,stale,first_epoch=-float('inf'),0,0
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        best,stale,first_epoch=state['best'],state['stale'],state['epoch']+1
        restore_rng(state['rng'],generator)
    for epoch in range(CONFIG['max_epochs']):
        if epoch<first_epoch:
            continue
        if stale>=CONFIG['patience']:
            break
        model.train()
        for values,mask,labels in training:
            values,mask=device_inputs(values,mask,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=criterion(model(values,mask),labels.to(device,non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite E3 loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],CONFIG['clip'],error_if_nonfinite=True)
            optimizer.step()
        auc=float(roc_auc_score(y[part['validation']],predict(model,validation,device)))
        if auc>best:
            best,stale=auc,0
            atomic_save({'model':model.state_dict(),'epoch':epoch,'validation_auroc':auc},directory/'best.pt')
        else:
            stale+=1
        atomic_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'stale':stale,'rng':rng_state(generator)},latest)
        print('E3_EPOCH',arm,'fold',fold,'seed',seed,'epoch',epoch,'validation_auroc',round(auc,5),flush=True)
    selected=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected['model'])
    validation_probability=predict(model,validation,device)
    thresholds={name:select_threshold(y[part['validation']],validation_probability,target) for name,target in [('threshold',None),('recall80_threshold',.8),('recall90_threshold',.9)]}
    test_probability=predict(model,make_loader(dataset(part['test']),False),device)
    result=score(y[part['test']],test_probability,thresholds['threshold'])
    result.update(selected_epoch=int(selected['epoch']),validation_auroc=float(selected['validation_auroc']))
    for label,indices,probabilities in [('validation',part['validation'],validation_probability),('test',part['test'],test_probability)]:
        output=frame.iloc[indices][['stay_id','subject_id','fold']].copy()
        output['y_true'],output['y_prob']=y[indices],probabilities
        for name,value in thresholds.items():
            output[name]=value
        temporary=directory/f'{label}.parquet.part'
        output.to_parquet(temporary,index=False)
        temporary.replace(directory/f'{label}.parquet')
    (directory/'metrics.json').write_text(json.dumps(result,indent=2))
    if result['auroc']>.98:
        raise RuntimeError('E3 test AUROC exceeds stop threshold')
    print('E3_CASE_COMPLETE',arm,fold,seed,flush=True)


def average_modalities(frames):
    ordered=[frame.sort_values('subject_id').reset_index(drop=True) for frame in frames]
    first=ordered[0].copy()
    for other in ordered[1:]:
        align_patients(first,other)
    first['y_prob']=np.mean([frame.y_prob.to_numpy() for frame in ordered],axis=0)
    return first


def late_fusion(root):
    for seed in range(3):
        rows=[]
        for fold in range(5):
            sets={name:average_modalities([pd.read_parquet(P/f'results/deep/e2/{modality}/fold{fold}/seed{seed}/{name}.parquet') for modality in MODALITIES]) for name in ('test','validation')}
            validation=sets['validation']
            for name,target in [('threshold',None),('recall80_threshold',.8),('recall90_threshold',.9)]:
                value=select_threshold(validation.y_true,validation.y_prob,target)
                for data in sets.values():
                    data[name]=value
            directory=root/'late_average'/f'fold{fold}'/f'seed{seed}'
            directory.mkdir(parents=True,exist_ok=True)
            for name,data in sets.items():
                data.to_parquet(directory/f'{name}.parquet',index=False)
            result=score(sets['test'].y_true,sets['test'].y_prob,sets['test'].threshold)
            (directory/'metrics.json').write_text(json.dumps(result,indent=2))
            rows.append(sets['test'])
        combined=pd.concat(rows).sort_values('subject_id')
        if len(combined)!=6225 or combined.subject_id.duplicated().any():
            raise RuntimeError('Late fusion OOF coverage mismatch')
        combined.to_parquet(P/f'results/oof/e3__late_average__seed{seed}.parquet',index=False)


def main():
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('E3 requires an allocated bf16 GPU')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if int(os.getxattr(P,'ceph.dir.rbytes'))>=90_000_000_000:
        raise RuntimeError('SSD use exceeds the 90 GB stop threshold')
    root=P/'results/deep/e3'
    root.mkdir(parents=True,exist_ok=True)
    e2=json.loads((P/'results/deep/e2/complete.json').read_text())
    if not e2['complete']:
        raise RuntimeError('E2 must be complete')
    raw=P/'data/processed/main'
    digest=hashlib.sha256(Path(__file__).read_bytes()+inspect.getsource(ConcatenationFusion).encode()+json.dumps(CONFIG,sort_keys=True).encode()+e2['signature'].encode())
    for file in ('cohort.parquet','partitions.npz'):
        digest.update((raw/file).read_bytes())
    for name in ('prepare_fold','make_loader','seed_all','rng_state','restore_rng','atomic_save'):
        digest.update(inspect.getsource(globals()[name]).encode())
    for modality in MODALITIES:
        for fold in range(5):
            for seed in range(3):
                digest.update((P/f'results/deep/e2/{modality}/fold{fold}/seed{seed}/encoder.pt').read_bytes())
    signature=digest.hexdigest()
    lock=root/'signature.txt'
    if lock.exists() and lock.read_text().strip()!=signature:
        raise RuntimeError('E3 signature changed after training began')
    lock.write_text(signature+'\n')
    (root/'config.json').write_text(json.dumps(CONFIG,indent=2))
    frame=pd.read_parquet(raw/'cohort.parquet')
    parts=np.load(raw/'partitions.npz',allow_pickle=False)
    cxr_indices=np.load(P/'data/processed/cxr/main_indices.npy',allow_pickle=False)
    for fold in range(5):
        part={key:parts[f'fold{fold}_{key}'] for key in ('train','validation','test')}
        features,valid=prepare_fold(P/'results/deep/e2',frame,part['train'],fold)
        for arm in CONFIG['arms']:
            for seed in CONFIG['seeds']:
                fit_case(root,frame,part,features,valid,cxr_indices,arm,fold,seed)
    for arm in CONFIG['arms']:
        for seed in CONFIG['seeds']:
            combined=pd.concat([pd.read_parquet(root/arm/f'fold{fold}'/f'seed{seed}'/'test.parquet') for fold in range(5)]).sort_values('subject_id')
            if len(combined)!=len(frame) or combined.subject_id.duplicated().any():
                raise RuntimeError('E3 OOF coverage mismatch')
            combined.to_parquet(P/f'results/oof/e3__{arm}__seed{seed}.parquet',index=False)
    late_fusion(root)
    (root/'complete.json').write_text(json.dumps({'complete':True,'signature':signature,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}))
    print('E3_FUSION_COMPLETE',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        reason=str(error) if isinstance(error,RuntimeError) else type(error).__name__
        (P/'reports/BLOCKED.md').write_text(f'E3 stopped: {reason}. Inspect before resuming.\n')
        print('E3_STOPPED',reason,flush=True)
        raise SystemExit(1)
