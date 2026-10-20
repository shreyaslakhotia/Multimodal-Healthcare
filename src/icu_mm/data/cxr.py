"""Frozen non-MIMIC CXR feature cache; CPU extraction avoids GPU queue delays."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Resize,CenterCrop,InterpolationMode
from .cohort import select_image_paths

P=Path('/projects/icumultimodal')
ARCHIVE=Path('/projects/icumultimodalarchive/zips/mimiccxr.zip')
WEIGHTS='densenet121-res224-chex'


def preprocess_image(image):
    image=image.convert('L')
    image=Resize(224,interpolation=InterpolationMode.BILINEAR)(image)
    pixels=np.asarray(CenterCrop(224)(image),dtype=np.uint8).copy()
    normalized=(pixels.astype(np.float32)/255.*2.-1.)*1024.
    return torch.from_numpy(normalized[None]),pixels


class ArchiveImages(Dataset):
    def __init__(self,members,indices):
        self.members,self.indices=members,indices
        self.archive=None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,position):
        if self.archive is None:
            self.archive=zipfile.ZipFile(ARCHIVE)
        index=self.indices[position]
        with Image.open(io.BytesIO(self.archive.read(self.members[index]))) as image:
            tensor,pixels=preprocess_image(image)
        return index,tensor,pixels


def audit_patient_paths(cohort):
    manifest=json.loads((P/'results/acquisition/cxrmetadata.json').read_text())
    metadata=pd.concat([pd.read_csv(P/f'raw/mimiccxr/metadata{i}.csv',usecols=['subject_id','AP','PA']) for i in range(len(manifest))])
    linked=cohort[['subject_id','selected_image']].merge(metadata,on='subject_id',validate='one_to_one')
    if len(linked)!=len(cohort):
        raise RuntimeError('Missing metadata in CXR linkage audit')
    mismatches=unknown=0
    for row in linked.itertuples(index=False):
        path,_,_=select_image_paths(row.AP,row.PA)
        match=re.search(r'(?:^|/)p(\d{8})/s\d+/',path.replace('\\','/')) if path else None
        if match is None:
            unknown+=1
        elif int(match.group(1))!=int(row.subject_id) or Path(path).name!=Path(row.selected_image).name:
            mismatches+=1
    if mismatches or unknown:
        raise RuntimeError(f'CXR patient/path audit failed: mismatches={mismatches}, unknown={unknown}')
    return {'patients_checked':len(linked),'mismatches':mismatches,'unparsed':unknown}


def array(path,shape,dtype):
    if path.exists():
        result=np.lib.format.open_memmap(path,mode='r+')
        if result.shape!=shape or result.dtype!=np.dtype(dtype):
            raise RuntimeError('Existing CXR cache array has incompatible shape/type')
        return result
    return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('CXR extraction requires a Slurm allocation')
    import torchxrayvision as xrv
    torch.set_num_threads(6)
    cohort=pd.read_parquet(P/'data/processed/main/cohort.parquet')
    linkage=audit_patient_paths(cohort)
    cache=P/'data/processed/cxr'
    cache.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        members=sorted(set(cohort.selected_image)|set(cohort.alternate_image),key=lambda name:archive.getinfo(name).header_offset)
    digest=hashlib.sha256(('\n'.join(members)+WEIGHTS+'gray_resize224_centercrop_bilinear_v1').encode()).hexdigest()
    manifest_path=cache/'manifest.json'
    old=None
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['signature']!=digest:
            raise RuntimeError('CXR cache signature changed; do not overwrite a completed experiment')
        if old.get('complete'):
            print('CXR_CACHE_ALREADY_COMPLETE',flush=True)
            return
    # Explicit cache_dir is essential: XRV's default is ~/.torchxrayvision,
    # independent of TORCH_HOME. Never write weights to HOME.
    model=xrv.models.DenseNet(weights=WEIGHTS,cache_dir=str(P/'.tmp/torch/xrv')).cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint=Path(model.weights_filename_local)
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if old is not None and old.get('weights_sha256')!=checkpoint_hash:
        raise RuntimeError('CXR checkpoint weights changed during a resumable extraction')
    info={'signature':digest,'weights':WEIGHTS,'weights_sha256':checkpoint_hash,
          'unique_images':len(members),'cohort_rows':len(cohort),'complete':False,
          'patient_linkage_audit':linkage,'device':'cpu','image_scale':[-1024,1024]}
    manifest_path.write_text(json.dumps(info,indent=2))
    count=len(members)
    tokens=array(cache/'tokens.npy',(count,49,1024),np.float16)
    pooled=array(cache/'pooled.npy',(count,1024),np.float16)
    pixels=array(cache/'images.npy',(count,224,224),np.uint8)
    done_path=cache/'done.npy'
    done=np.load(done_path,allow_pickle=False) if done_path.exists() else np.zeros(count,dtype=bool)
    if done.shape!=(count,):
        raise RuntimeError('CXR checkpoint length changed')
    index={member:i for i,member in enumerate(members)}
    np.save(cache/'main_indices.npy',[index[name] for name in cohort.selected_image],allow_pickle=False)
    np.save(cache/'alternate_indices.npy',[index[name] for name in cohort.alternate_image],allow_pickle=False)
    pending=np.flatnonzero(~done).tolist()
    loader=DataLoader(ArchiveImages(members,pending),batch_size=16,num_workers=2,
                      multiprocessing_context='spawn',persistent_workers=True,shuffle=False)
    started=time.monotonic()
    processed=0
    with torch.inference_mode():
        for indices,inputs,images in loader:
            feature=torch.relu(model.features(inputs))
            if feature.shape[1:]!=(1024,7,7):
                raise RuntimeError('Unexpected DenseNet spatial feature shape')
            token=feature.flatten(2).transpose(1,2).numpy().astype(np.float16)
            mean=feature.mean(dim=(2,3)).numpy().astype(np.float16)
            if not np.isfinite(token).all() or not np.isfinite(mean).all():
                raise RuntimeError('Nonfinite frozen CXR features')
            position=indices.numpy()
            tokens[position]=token
            pooled[position]=mean
            pixels[position]=images.numpy()
            for matrix in (tokens,pooled,pixels):
                matrix.flush()
            done[position]=True
            temporary=done_path.with_suffix('.npy.tmp')
            with temporary.open('wb') as stream:
                np.save(stream,done,allow_pickle=False)
            temporary.replace(done_path)
            processed+=len(position)
            rate=processed/max(time.monotonic()-started,1e-6)
            remaining=int((~done).sum()/max(rate,1e-6))
            print('CXR_PROGRESS',int(done.sum()),'of',count,'images_per_second',round(rate,2),'eta_seconds',remaining,flush=True)
    info['complete']=True
    manifest_path.write_text(json.dumps(info,indent=2))
    print('CXR_FROZEN_CACHE_READY',count,flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        reason=str(error) if isinstance(error,RuntimeError) else type(error).__name__
        (P/'reports/BLOCKED.md').write_text(f'CXR extraction stopped: {reason}. No automatic retries.\n')
        print('CXR_STOPPED',reason,flush=True)
        raise SystemExit(1)
