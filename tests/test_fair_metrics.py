import numpy as np
import pandas as pd
from icu_mm.fair_metrics import paired_evidence


def test_identical_models_are_reported_unresolved_and_each_fold_is_retained():
    frame=pd.DataFrame({'subject_id':np.arange(100),'fold':np.repeat(np.arange(5),20),
                        'y_true':np.tile([0,1],50),'y_prob':np.tile([.2,.8],50)})
    result=paired_evidence(frame,frame,iterations=20)
    assert result['auroc_status']=='unresolved'
    assert result['auroc_zero_folds']==5
    assert all(result[f'auroc_fold{fold}_delta']==0 for fold in range(5))
