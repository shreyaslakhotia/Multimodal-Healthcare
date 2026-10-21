import pandas as pd
import pytest
from icu_mm.evaluate_e2 import average_seeds


def test_seed_averaging_aligns_unique_patients_and_rejects_changed_labels():
    first = pd.DataFrame({'subject_id':[1,2], 'fold':[0,0], 'y_true':[0,1], 'y_prob':[.1,.8]})
    second = first.iloc[::-1].copy()
    second['y_prob'] = [.6,.3]
    result = average_seeds([first,second])
    assert result.y_prob.tolist() == pytest.approx([.2,.7])
    second.loc[second.subject_id == 1, 'y_true'] = 1
    with pytest.raises(ValueError):
        average_seeds([first,second])
