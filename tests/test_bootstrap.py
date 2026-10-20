import numpy as np
import pytest
from icu_mm.bootstrap import paired_bootstrap,patient_intervals


def test_paired_identical_predictions_have_exactly_zero_interval():
    y = np.arange(200) % 4 == 0
    p = np.random.default_rng(1).random(200)
    result = paired_bootstrap(y, np.arange(200), p, p, iterations=100)
    for metric in ('auroc', 'auprc'):
        assert result[metric] == {'delta': 0., 'lower': 0., 'upper': 0.}


def test_seed_copies_are_not_independent_patients():
    with pytest.raises(ValueError, match='one prediction per patient'):
        paired_bootstrap([0, 1, 0, 1], [5, 6, 5, 6], [.1, .9, .2, .8], [.2, .8, .3, .7])


def test_perfect_predictions_have_unit_intervals():
    result=patient_intervals([0,1,0,1],range(4),[0.,1.,0.,1.],iterations=100)
    assert result['auroc']=={'lower':1.,'upper':1.}
    assert result['auprc']=={'lower':1.,'upper':1.}
