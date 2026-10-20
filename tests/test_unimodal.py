import torch
from icu_mm.models import masked_pool,UnimodalModel


def test_masked_pool_excludes_unobserved_hours_and_handles_empty_stream():
    tokens=torch.tensor([[[1.,2.],[100.,200.]],[[3.,4.],[5.,6.]]])
    valid=torch.tensor([[True,False],[False,False]])
    pooled=masked_pool(tokens,valid)
    torch.testing.assert_close(pooled,torch.tensor([[1.,2.],[0.,0.]]))


def test_unimodal_head_has_finite_backward_for_context_and_temporal_input():
    for modality,values,valid in [('context',torch.randn(3,22),torch.ones(3,1,dtype=torch.bool)),
                                 ('labs',torch.randn(3,48,22),torch.ones(3,48,dtype=torch.bool))]:
        model=UnimodalModel(modality,22)
        logits=model(values,valid)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.,0.]))
        loss.backward()
        assert logits.shape==(3,) and torch.isfinite(loss)


def test_worker_initialization_cannot_change_resumed_shuffle_order():
    from icu_mm.train_unimodal import make_loader
    generator=torch.Generator().manual_seed(55)
    loader=make_loader(list(range(100)),True,generator)
    list(loader.sampler)  # Simulate one completed epoch.
    saved=generator.get_state()
    torch.empty((),dtype=torch.int64).random_(generator=loader.generator)
    expected=list(loader.sampler)
    resumed=torch.Generator()
    resumed.set_state(saved)
    second=make_loader(list(range(100)),True,resumed)
    assert loader.generator is not generator
    assert list(second.sampler)==expected
