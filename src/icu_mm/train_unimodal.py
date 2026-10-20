"""Fixed E2 encoders: three seeds, five folds, restartable GPU training."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,RandomSampler
from sklearn.metrics import roc_auc_score
from .context import ContextEncoder
from .preprocessing import HourlyNormalizer
from .models import UnimodalModel,ModalityEncoders,SequenceEncoder,masked_pool,sinusoidal,MODALITIES
from .metrics import score,select_threshold

P=Path('/projects/icumultimodal')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state(generator):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all(),'loader':generator.get_state()}


def restore_rng(state,generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])


def atomic_save(value,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


class Inputs(Dataset):
    def __init__(self,modality,indices,features,valid,labels,cxr_indices=None):
        self.modality,self.indices=modality,np.asarray(indices)
        self.features,self.valid,self.labels=features,valid,labels
        self.cxr_indices=cxr_indices
        self.tokens=None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        row=self.indices[index]
        if self.modality=='cxr':
            if self.tokens is None:
                self.tokens=np.load(P/'data/processed/cxr/tokens.npy',mmap_mode='r')
            values=np.array(self.tokens[self.cxr_indices[row]],dtype=np.float32,copy=True)
            valid=np.ones(49,dtype=bool)
        else:
            values=np.array(self.features[row],dtype=np.float32,copy=True)
            valid=self.valid[row]
        return torch.from_numpy(values),torch.as_tensor(valid,dtype=torch.bool),torch.tensor(self.labels[row],dtype=torch.float32)


def make_loader(dataset,train,generator=None):
    # Sampling has a separate RNG from worker initialization. Recreating
    # workers on resume must not consume the saved shuffle/dropout state.
    arguments={'batch_size':64,'shuffle':False,'num_workers':2 if train else 0,
               'pin_memory':True}
    if train:
        arguments.update(multiprocessing_context='spawn',persistent_workers=True,
                         sampler=RandomSampler(dataset,generator=generator),
                         generator=torch.Generator().manual_seed(9137))
    return DataLoader(dataset,**arguments)


def predict(model,loader,device):
    model.eval()
    probabilities=[]
    with torch.inference_mode():
        for values,valid,_ in loader:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(values.to(device,non_blocking=True),valid.to(device,non_blocking=True))
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probabilities)


def prepare_fold(root,frame,train,fold):
    path=root/f'preprocessing_fold{fold}.joblib'
    train_hash=hashlib.sha256(np.asarray(train,dtype=np.int64).tobytes()).hexdigest()
    raw=P/'data/processed/main'
    if path.exists():
        prep=joblib.load(path)
        if prep['train_hash']!=train_hash:
            raise RuntimeError('Deep preprocessing training partition changed')
    else:
        prep={'train_hash':train_hash,'context':ContextEncoder().fit(frame.iloc[train])}
        for modality in ('vitals','labs'):
            values=np.load(raw/f'{modality}.npy',mmap_mode='r')
            mask=np.load(raw/f'{modality}_mask.npy',mmap_mode='r')
            prep[modality]=HourlyNormalizer().fit(values[train],mask[train])
        joblib.dump(prep,path)
    features={'context':prep['context'].transform(frame)}
    valid={'context':np.ones((len(frame),1),dtype=bool)}
    for modality in ('vitals','labs'):
        values=np.load(raw/f'{modality}.npy',mmap_mode='r')
        mask=np.load(raw/f'{modality}_mask.npy',mmap_mode='r')
        normalized=prep[modality].transform(values,mask)
        features[modality]=np.concatenate([normalized,mask.astype(np.float32)],axis=2)
        valid[modality]=mask.any(axis=2)
    return features,valid


def fit_case(root,frame,part,features,valid,cxr_indices,modality,fold,seed):
    directory=root/modality/f'fold{fold}'/f'seed{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    if all((directory/name).exists() for name in ('test.parquet','validation.parquet','metrics.json','encoder.pt')):
        previous=json.loads((directory/'metrics.json').read_text())
        if previous['auroc']>.98 or (modality!='cxr' and previous['auroc']>.90):
            raise RuntimeError('Persisted E2 result still exceeds leakage gate; do not bypass on resume')
        return
    seed_all(seed)
    device=torch.device('cuda')
    model=UnimodalModel(modality,features['context'].shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-2)
    y=frame.label.to_numpy()
    training_y=y[part['train']]
    positive=int(training_y.sum())
    if positive==0 or positive==len(training_y):
        raise RuntimeError('Training portion lacks both outcome classes')
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(training_y)-positive)/positive,device=device))
    generator=torch.Generator().manual_seed(42+seed+100*fold)
    def dataset(indices):
        return Inputs(modality,indices,features.get(modality),valid.get(modality),y,cxr_indices)
    training=make_loader(dataset(part['train']),True,generator)
    validation=make_loader(dataset(part['validation']),False)
    best,stale,first_epoch=-float('inf'),0,0
    latest=directory/'latest.pt'
    if latest.exists():
        # These are trusted checkpoints written by this program on private SSD.
        state=torch.load(latest,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        best,stale,first_epoch=state['best'],state['stale'],state['epoch']+1
        restore_rng(state['rng'],generator)
    for epoch in range(first_epoch,50):
        if stale>=10:
            break
        model.train()
        for values,mask,labels in training:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(values.to(device,non_blocking=True),mask.to(device,non_blocking=True))
                loss=criterion(logits,labels.to(device,non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite deep training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        probability=predict(model,validation,device)
        auc=float(roc_auc_score(y[part['validation']],probability))
        if auc>best:
            best,stale=auc,0
            atomic_save({'model':model.state_dict(),'epoch':epoch,'validation_auroc':auc},directory/'best.pt')
        else:
            stale+=1
        atomic_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'epoch':epoch,'best':best,'stale':stale,'rng':rng_state(generator)},latest)
        print('E2_EPOCH',modality,'fold',fold,'seed',seed,'epoch',epoch,'validation_auroc',round(auc,5),flush=True)
    selected=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected['model'])
    validation_probability=predict(model,validation,device)
    threshold=select_threshold(y[part['validation']],validation_probability)
    recall80=select_threshold(y[part['validation']],validation_probability,.8)
    recall90=select_threshold(y[part['validation']],validation_probability,.9)
    # Selection is finished before the one locked test prediction.
    test_probability=predict(model,make_loader(dataset(part['test']),False),device)
    result=score(y[part['test']],test_probability,threshold)
    result.update(selected_epoch=int(selected['epoch']),validation_auroc=float(selected['validation_auroc']))
    for label,indices,probabilities in [('validation',part['validation'],validation_probability),('test',part['test'],test_probability)]:
        output=frame.iloc[indices][['stay_id','subject_id','fold']].copy()
        output['y_true']=y[indices]
        output['y_prob']=probabilities
        output['threshold']=threshold
        output['recall80_threshold']=recall80
        output['recall90_threshold']=recall90
        temporary=directory/f'{label}.parquet.part'
        output.to_parquet(temporary,index=False)
        temporary.replace(directory/f'{label}.parquet')
    atomic_save(model.encoder.state_dict(),directory/'encoder.pt')
    (directory/'metrics.json').write_text(json.dumps(result,indent=2))
    if result['auroc']>.98 or (modality!='cxr' and result['auroc']>.90):
        raise RuntimeError('Deep test AUROC exceeds a prespecified leakage gate')
    print('E2_CASE_COMPLETE',modality,fold,seed,flush=True)


def main():
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available():
        raise RuntimeError('E2 requires an allocated GPU')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Allocated GPU does not support required bf16')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    root=P/'results/deep/e2'
    root.mkdir(parents=True,exist_ok=True)
    raw=P/'data/processed/main'
    frame=pd.read_parquet(raw/'cohort.parquet')
    parts=np.load(raw/'partitions.npz',allow_pickle=False)
    cxr_indices=np.load(P/'data/processed/cxr/main_indices.npy',allow_pickle=False)
    manifest=json.loads((P/'data/processed/cxr/manifest.json').read_text())
    if not manifest['complete'] or len(cxr_indices)!=len(frame):
        raise RuntimeError('Frozen CXR cache incomplete or misaligned')
    digest=hashlib.sha256(Path(__file__).read_bytes()+(raw/'cohort.parquet').read_bytes()+(raw/'partitions.npz').read_bytes())
    for definition in (UnimodalModel,ModalityEncoders,SequenceEncoder,masked_pool,sinusoidal,ContextEncoder,HourlyNormalizer,select_threshold,score):
        digest.update(inspect.getsource(definition).encode())
    digest.update((P/'results/cohort/features.json').read_bytes())
    digest.update((manifest['signature']+manifest['weights_sha256']).encode())
    lock=root/'signature.txt'
    signature=digest.hexdigest()
    if lock.exists() and lock.read_text().strip()!=signature:
        raise RuntimeError('E2 experiment signature changed after training began')
    lock.write_text(signature+'\n')
    for fold in range(5):
        part={key:parts[f'fold{fold}_{key}'] for key in ('train','validation','test')}
        features,valid=prepare_fold(root,frame,part['train'],fold)
        for modality in MODALITIES:
            for seed in (0,1,2):
                fit_case(root,frame,part,features,valid,cxr_indices,modality,fold,seed)
    for modality in MODALITIES:
        for seed in (0,1,2):
            predictions=pd.concat([pd.read_parquet(root/modality/f'fold{fold}'/f'seed{seed}'/'test.parquet') for fold in range(5)]).sort_values('subject_id')
            if len(predictions)!=len(frame) or predictions.subject_id.duplicated().any():
                raise RuntimeError('Deep OOF coverage mismatch')
            predictions.to_parquet(P/f'results/oof/e2__{modality}__seed{seed}.parquet',index=False)
    (root/'complete.json').write_text(json.dumps({'complete':True,'signature':signature,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}))
    print('E2_UNIMODAL_COMPLETE',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        reason=str(error) if isinstance(error,RuntimeError) else type(error).__name__
        (P/'reports/BLOCKED.md').write_text(f'E2 stopped: {reason}. Inspect before resuming.\n')
        print('E2_STOPPED',reason,flush=True)
        raise SystemExit(1)
