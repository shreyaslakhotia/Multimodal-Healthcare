import numpy as np
import pandas as pd
import pytest
import torch
from icu_mm.fusion import ConcatenationFusion
from icu_mm.train_fusion import average_modalities
from test_models import synthetic_inputs


def test_frozen_encoders_stay_deterministic_and_only_head_updates():
    torch.manual_seed(7)
    model=ConcatenationFusion(22,frozen=True).train()
    values=synthetic_inputs()
    valid={'context':torch.ones(2,1,dtype=torch.bool),'vitals':torch.ones(2,48,dtype=torch.bool),'labs':torch.ones(2,48,dtype=torch.bool),'cxr':torch.ones(2,49,dtype=torch.bool)}
    first=model.encode(values,valid)
    second=model.encode(values,valid)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    before={name:p.detach().clone() for name,p in model.encoders.named_parameters()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    output=model(values,valid)
    output.sum().backward(); optimizer.step()
    assert all(p.grad is None for p in model.encoders.parameters())
    assert all(torch.equal(before[name],p) for name,p in model.encoders.named_parameters())
    assert model.head[0].weight.grad is not None


def test_joint_fusion_receives_finite_encoder_gradients():
    model=ConcatenationFusion(22).eval()
    values=synthetic_inputs()
    valid={'context':torch.ones(2,1,dtype=torch.bool),'vitals':torch.ones(2,48,dtype=torch.bool),'labs':torch.ones(2,48,dtype=torch.bool),'cxr':torch.ones(2,49,dtype=torch.bool)}
    result=model(values,valid)
    assert result.shape==(2,) and torch.isfinite(result).all()
    result.sum().backward()
    assert all(any(p.grad is not None and torch.isfinite(p.grad).all() for p in getattr(model.encoders,name).parameters()) for name in ('context','vitals','labs','cxr'))


def test_late_fusion_aligns_patients_and_rejects_mismatched_folds():
    a=pd.DataFrame({'subject_id':[2,1],'fold':[0,0],'y_true':[1,0],'y_prob':[.8,.2]})
    b=a.iloc[::-1].copy(); b['y_prob']=[.4,.6]
    result=average_modalities([a,b])
    np.testing.assert_allclose(result.y_prob,[.3,.7])
    b.loc[b.subject_id==1,'fold']=1
    with pytest.raises(ValueError): average_modalities([a,b])
