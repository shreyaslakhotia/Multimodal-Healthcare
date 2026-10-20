import numpy as np
import pandas as pd
from icu_mm.context import ContextEncoder


def test_unknown_context_uses_explicit_slots_without_growing_vocabulary():
    train = pd.DataFrame({'age':[30.,50.],'sex':[0,1],'admission_type':['A','A'],'first_careunit':['ICU1','ICU2']})
    heldout = pd.DataFrame({'age':[1000.],'sex':[1],'admission_type':['NEW'],'first_careunit':['NEW']})
    encoder = ContextEncoder().fit(train)
    names = encoder.feature_names_.copy()
    result = encoder.transform(heldout)
    assert encoder.feature_names_ == names
    assert encoder.mean_ == 40
    assert result[0,names.index('admission_type__other')] == 1
    assert result[0,names.index('first_careunit__other')] == 1
    assert result.shape[1] == len(names)
    assert result[0,0] == 96
