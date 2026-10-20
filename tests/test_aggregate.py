import pandas as pd
import pytest
from icu_mm.aggregate import align_patients


def test_paired_comparison_aligns_ids_and_rejects_fold_or_label_mismatch():
    first=pd.DataFrame({'subject_id':[1,2],'fold':[0,1],'y_true':[0,1],'y_prob':[.1,.8]})
    second=first.iloc[::-1].copy()
    paired=align_patients(first,second)
    assert (paired.y_prob_first==paired.y_prob_second).all()
    second.loc[second.subject_id==1,'fold']=4
    with pytest.raises(ValueError):
        align_patients(first,second)
