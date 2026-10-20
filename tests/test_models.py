import torch
from icu_mm.models import CrossAttentionFusion


def synthetic_inputs(batch=2):
    return {'context': torch.randn(batch, 22), 'vitals': torch.randn(batch, 48, 16),
            'labs': torch.randn(batch, 48, 22), 'cxr': torch.randn(batch, 49, 1024)}


def test_missing_cxr_cannot_influence_visible_prediction():
    torch.manual_seed(5)
    model = CrossAttentionFusion(22).eval()
    inputs = synthetic_inputs()
    present = torch.ones(2, 4, dtype=torch.bool)
    present[:, 3] = False
    with torch.no_grad():
        first = model(inputs, present)
        inputs['cxr'].fill_(float('nan'))
        second = model(inputs, present)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(second).all()


def test_context_only_after_dropout_has_finite_logits_and_gradients():
    model = CrossAttentionFusion(22, modality_dropout=1.).train()
    output = model(synthetic_inputs())
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
