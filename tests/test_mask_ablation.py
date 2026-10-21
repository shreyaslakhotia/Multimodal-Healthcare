import numpy as np
from icu_mm.mask_ablation import variant_inputs


def test_values_cannot_influence_mask_only_and_masks_cannot_enter_values_channels():
    values=np.arange(48*8,dtype=np.float32).reshape(1,48,8)
    masks=np.ones_like(values)
    both=np.concatenate([values,masks],axis=-1)
    first=variant_inputs(both,'mask_only')
    both[:,:,:8]=-100
    np.testing.assert_array_equal(first,variant_inputs(both,'mask_only'))
    selected=variant_inputs(both,'values_only')
    both[:,:,8:]=0
    np.testing.assert_array_equal(selected,variant_inputs(both,'values_only'))
    assert first.shape==selected.shape==(1,48,8)
