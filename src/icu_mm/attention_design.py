"""EHR-first E4 architecture with explicit active modalities and missingness."""
import torch
from torch import nn
from .models import MODALITIES,ModalityEncoders,masked_pool


class ActiveEncoders(nn.Module):
    def __init__(self,context_dim,modalities):
        super().__init__()
        if len(modalities)<2 or len(set(modalities))!=len(modalities) or any(name not in MODALITIES for name in modalities):
            raise ValueError('Expected at least two distinct known modalities')
        self.modalities=tuple(modalities)
        reference=ModalityEncoders(context_dim)
        self.streams=nn.ModuleDict({name:getattr(reference,name) for name in self.modalities})

    def encode(self,values,valid,present=None):
        first=values[self.modalities[0]]
        batch=first.shape[0]
        if present is None:
            present=torch.ones(batch,len(self.modalities),dtype=torch.bool,device=first.device)
        else:
            present=present.to(device=first.device,dtype=torch.bool)
        if present.shape!=(batch,len(self.modalities)):
            raise ValueError('Incorrect modality presence shape')
        streams,masks=[],[]
        for index,name in enumerate(self.modalities):
            value=values[name]
            observed=valid[name].bool() & present[:,index:index+1]
            safe=torch.where(present[:,index].reshape([batch]+[1]*(value.ndim-1)),value,0.)
            encoded=self.streams[name](safe)
            streams.append(encoded.unsqueeze(1) if name=='context' else encoded)
            masks.append(observed)
        return streams,masks


class EHRConcatenationFusion(ActiveEncoders):
    def __init__(self,context_dim,modalities=('context','vitals','labs')):
        super().__init__(context_dim,modalities)
        self.head=nn.Sequential(nn.Linear(128*len(modalities),128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,1))

    def forward(self,values,valid,present=None):
        streams,masks=self.encode(values,valid,present)
        return self.head(torch.cat([masked_pool(stream,mask) for stream,mask in zip(streams,masks)],dim=1)).squeeze(-1)


class ActiveCrossBlock(nn.Module):
    def __init__(self,count):
        super().__init__()
        self.attention=nn.ModuleList([nn.MultiheadAttention(128,4,dropout=.1,batch_first=True) for _ in range(count)])
        self.norm1=nn.ModuleList([nn.LayerNorm(128) for _ in range(count)])
        self.norm2=nn.ModuleList([nn.LayerNorm(128) for _ in range(count)])
        self.ffn=nn.ModuleList([nn.Sequential(nn.Linear(128,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,128),nn.Dropout(.1)) for _ in range(count)])
        self.dropout=nn.Dropout(.1)

    def forward(self,streams,masks,log_attention=False):
        outputs,diagnostics=[],{}
        for query in range(len(streams)):
            others=[index for index in range(len(streams)) if index!=query]
            keys=torch.cat([streams[index] for index in others],dim=1)
            key_valid=torch.cat([masks[index] for index in others],dim=1)
            any_other=key_valid.any(dim=1,keepdim=True)
            keys=torch.cat([keys,torch.zeros_like(keys[:,:1])],dim=1)
            allowed=torch.cat([key_valid,~any_other],dim=1)
            updated,weights=self.attention[query](streams[query],keys,keys,key_padding_mask=~allowed,need_weights=log_attention)
            updated=self.norm1[query](streams[query]+self.dropout(updated))
            outputs.append(self.norm2[query](updated+self.ffn[query](updated)))
            if log_attention:
                valid_query=masks[query].to(weights.dtype)
                usable=masks[query].any(dim=1)&any_other.squeeze(1)
                start=0
                for other in others:
                    length=streams[other].shape[1]
                    mass=weights[:,:,start:start+length].sum(dim=2)
                    per_patient=(mass*valid_query).sum(dim=1)/valid_query.sum(dim=1).clamp_min(1)
                    diagnostics[(query,other)]={'mass_sum':per_patient[usable].sum().detach(),'patients':usable.sum().detach()}
                    start+=length
        return outputs,diagnostics


class EHRFirstCrossAttention(ActiveEncoders):
    def __init__(self,context_dim,modalities=('context','vitals','labs'),modality_dropout=0.):
        super().__init__(context_dim,modalities)
        if not 0<=modality_dropout<=1: raise ValueError('Invalid dropout probability')
        self.modality_dropout=modality_dropout
        self.tags=nn.Parameter(torch.randn(len(modalities),128)*.02)
        self.missing=nn.Parameter(torch.randn(len(modalities),128)*.02)
        self.blocks=nn.ModuleList([ActiveCrossBlock(len(modalities)) for _ in range(2)])
        self.head=nn.Sequential(nn.Linear(128*len(modalities),128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,1))

    def forward(self,values,valid,present=None,log_attention=False):
        first=values[self.modalities[0]]
        if present is None:
            present=torch.ones(first.shape[0],len(self.modalities),device=first.device,dtype=torch.bool)
        else: present=present.to(device=first.device,dtype=torch.bool).clone()
        if self.training and self.modality_dropout:
            for index,name in enumerate(self.modalities):
                if name!='context': present[:,index] &= torch.rand(len(first),device=first.device)>=self.modality_dropout
        streams,masks=self.encode(values,valid,present)
        streams=[stream+self.tags[index] for index,stream in enumerate(streams)]
        logs=[]
        for block in self.blocks:
            streams,diagnostics=block(streams,masks,log_attention)
            logs.append(diagnostics)
        pooled=[torch.where(mask.any(dim=1,keepdim=True),masked_pool(stream,mask),self.missing[index]) for index,(stream,mask) in enumerate(zip(streams,masks))]
        output=self.head(torch.cat(pooled,dim=1)).squeeze(-1)
        return (output,logs) if log_attention else output
