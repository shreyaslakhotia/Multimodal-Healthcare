"""E2 temporal values-only and mask-only controls on the locked partitions."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from .models import SequenceEncoder,masked_pool
from .train_unimodal import Inputs,prepare_fold,make_loader,predict,seed_all,atomic_save,rng_state,restore_rng
from .metrics import select_threshold,score

P=Path('/projects/icumultimodal')


class TemporalControl(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.encoder=SequenceEncoder(channels,48)
        self.head=nn.Linear(128,1)

    def forward(self,values,valid):
        return self.head(masked_pool(self.encoder(values),valid)).squeeze(-1)


def variant_inputs(features,variant):
    if features.shape[-1]%2:
        raise ValueError('Expected values followed by equally many masks')
    channels=features.shape[-1]//2
    if variant=='values_only': return features[:,:,:channels].copy()
    if variant=='mask_only': return features[:,:,channels:].copy()
    raise ValueError('Unknown mask ablation')


def fit_case(root,frame,part,features,valid,modality,fold,seed):
    directory=root/modality/f'fold{fold}'/f'seed{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    if all((directory/file).exists() for file in ('test.parquet','validation.parquet','best.pt','metrics.json')):
        if json.loads((directory/'metrics.json').read_text())['auroc']>.90:
            raise RuntimeError('Persisted temporal control exceeds EHR leakage gate')
        return
    seed_all(seed); device=torch.device('cuda'); y=frame.label.to_numpy()
    model=TemporalControl(features.shape[-1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    positives=y[part['train']].sum()
    if not 0<positives<len(part['train']): raise RuntimeError('Both classes required')
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(part['train'])-positives)/positives,device=device))
    generator=torch.Generator().manual_seed(42+seed+100*fold)
    def dataset(indices): return Inputs(modality,indices,features,valid,y)
    training=make_loader(dataset(part['train']),True,generator)
    validation=make_loader(dataset(part['validation']),False)
    latest=directory/'latest.pt'; best,stale,start=-float('inf'),0,0
    if latest.exists():
        checkpoint=torch.load(latest,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        best,stale,start=checkpoint['best'],checkpoint['stale'],checkpoint['epoch']+1
        restore_rng(checkpoint['rng'],generator)
    for epoch in range(start,50):
        if stale>=10: break
        model.train()
        for values,mask,labels in training:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=criterion(model(values.to(device),mask.to(device)),labels.to(device))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite ablation loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
        auc=float(roc_auc_score(y[part['validation']],predict(model,validation,device)))
        if auc>best:
            best,stale=auc,0
            atomic_save({'model':model.state_dict(),'epoch':epoch,'validation_auroc':auc},directory/'best.pt')
        else: stale+=1
        atomic_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'stale':stale,'rng':rng_state(generator)},latest)
        print('MASK_EPOCH',root.name,modality,fold,seed,epoch,round(auc,5),flush=True)
    selected=torch.load(directory/'best.pt',map_location='cpu',weights_only=True); model.load_state_dict(selected['model'])
    val_prob=predict(model,validation,device)
    thresholds={name:select_threshold(y[part['validation']],val_prob,target) for name,target in [('threshold',None),('recall80_threshold',.8),('recall90_threshold',.9)]}
    test_prob=predict(model,make_loader(dataset(part['test']),False),device)
    result=score(y[part['test']],test_prob,thresholds['threshold'])
    result.update(selected_epoch=int(selected['epoch']),validation_auroc=float(selected['validation_auroc']))
    for name,indices,probability in [('test',part['test'],test_prob),('validation',part['validation'],val_prob)]:
        output=frame.iloc[indices][['subject_id','stay_id','fold']].copy()
        output['y_true'],output['y_prob']=y[indices],probability
        for column,value in thresholds.items(): output[column]=value
        temporary=directory/f'{name}.part'; output.to_parquet(temporary,index=False); temporary.replace(directory/f'{name}.parquet')
    (directory/'metrics.json').write_text(json.dumps(result,indent=2))
    if result['auroc']>.90: raise RuntimeError('EHR ablation AUROC exceeds leakage gate')
    print('MASK_CASE_COMPLETE',root.name,modality,fold,seed,flush=True)


def main():
    if 'SLURM_JOB_ID' not in os.environ or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Temporal ablation requires an allocated bf16 GPU')
    torch.set_num_threads(2); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    if int(os.getxattr(P,'ceph.dir.rbytes'))>=90_000_000_000: raise RuntimeError('SSD stop threshold')
    root=P/'results/deep/e2_mask_ablation'; root.mkdir(parents=True,exist_ok=True)
    raw=P/'data/processed/main'; frame=pd.read_parquet(raw/'cohort.parquet'); parts=np.load(raw/'partitions.npz',allow_pickle=False)
    digest=hashlib.sha256(Path(__file__).read_bytes()+(raw/'partitions.npz').read_bytes()+(P/'results/deep/e2/complete.json').read_bytes())
    for item in (SequenceEncoder,masked_pool,Inputs,prepare_fold,make_loader,predict,seed_all,rng_state,restore_rng): digest.update(inspect.getsource(item).encode())
    signature=digest.hexdigest(); lock=root/'signature.txt'
    if lock.exists() and lock.read_text().strip()!=signature: raise RuntimeError('Ablation signature changed')
    lock.write_text(signature+'\n')
    for fold in range(5):
        part={key:parts[f'fold{fold}_{key}'] for key in ('train','validation','test')}
        features,valid=prepare_fold(P/'results/deep/e2',frame,part['train'],fold)
        for variant in ('values_only','mask_only'):
            for modality in ('vitals','labs'):
                selected=variant_inputs(features[modality],variant)
                for seed in range(3): fit_case(root/variant,frame,part,selected,valid[modality],modality,fold,seed)
    for variant in ('values_only','mask_only'):
        for modality in ('vitals','labs'):
            for seed in range(3):
                combined=pd.concat([pd.read_parquet(root/variant/modality/f'fold{fold}'/f'seed{seed}'/'test.parquet') for fold in range(5)]).sort_values('subject_id')
                if len(combined)!=len(frame) or combined.subject_id.duplicated().any(): raise RuntimeError('Ablation OOF mismatch')
                combined.to_parquet(P/f'results/oof/e2control__{variant}__{modality}__seed{seed}.parquet',index=False)
    (root/'complete.json').write_text(json.dumps({'complete':True,'signature':signature,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'pooling':'Original any-channel-observed hour validity retained in all arms; values-only removes explicit mask channels, not structural mask information',
      'attention':'Imputed tokens remain visible to self-attention; pooling excludes entirely unobserved hours'}))
    print('TEMPORAL_ABLATIONS_COMPLETE',flush=True)


if __name__=='__main__':
    try: main()
    except Exception as error:
        reason=str(error) if isinstance(error,RuntimeError) else type(error).__name__
        (P/'reports/BLOCKED.md').write_text(f'Temporal ablation stopped: {reason}. Inspect before resuming.\n')
        print('MASK_ABLATION_STOPPED',reason,flush=True); raise SystemExit(1)
