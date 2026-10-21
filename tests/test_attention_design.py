import torch
from icu_mm.attention_design import EHRFirstCrossAttention,EHRConcatenationFusion
from test_models import synthetic_inputs


def masks():
    return {'context':torch.ones(2,1,dtype=torch.bool),'vitals':torch.ones(2,48,dtype=torch.bool),'labs':torch.ones(2,48,dtype=torch.bool),'cxr':torch.ones(2,49,dtype=torch.bool)}


def test_ehr_models_neither_require_nor_use_cxr():
    values=synthetic_inputs(); valid=masks()
    for model in (EHRFirstCrossAttention(22).eval(),EHRConcatenationFusion(22).eval()):
        with torch.no_grad():
            first=model(values,valid)
            without={name:value for name,value in values.items() if name!='cxr'}
            second=model(without,valid)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        assert 'cxr' not in model.streams


def test_masked_cxr_is_inert_and_all_missing_uses_finite_null_attention():
    model=EHRFirstCrossAttention(22,('context','vitals','labs','cxr')).eval()
    values=synthetic_inputs(); valid=masks(); present=torch.ones(2,4,dtype=torch.bool); present[:,3]=False
    with torch.no_grad():
        first=model(values,valid,present)
        values['cxr'].fill_(float('nan'))
        second=model(values,valid,present)
        empty=model(values,valid,torch.zeros_like(present))
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert torch.isfinite(empty).all()


def test_attention_mass_is_descriptive_and_excludes_self_targets():
    model=EHRFirstCrossAttention(22).eval()
    with torch.no_grad():
        _,logs=model(synthetic_inputs(),masks(),log_attention=True)
    assert len(logs)==2
    for block in logs:
        for source in range(3):
            total=sum(record['mass_sum'] for (query,_),record in block.items() if query==source)
            torch.testing.assert_close(total,torch.tensor(2.),atol=1e-5,rtol=1e-5)
        assert all(source!=target for source,target in block)
