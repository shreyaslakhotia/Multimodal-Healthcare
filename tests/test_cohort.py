import pandas as pd
from icu_mm.data.cohort import landmark_eligibility, select_image_paths, resolve_member


def test_image_selection_does_not_fallback_when_first_ap_is_missing():
    selected, view, options = select_image_paths("['missing.jpg','present.jpg']", "['pa.jpg']")
    assert (selected, view) == ('missing.jpg','AP')
    assert resolve_member(selected, {'present.jpg','pa.jpg'}, {'present.jpg':['present.jpg'],'pa.jpg':['pa.jpg']}) is None
    assert select_image_paths('[]', "['pa.jpg']")[:2] == ('pa.jpg','PA')


def test_ambiguous_basename_is_rejected():
    assert resolve_member('elsewhere/a.jpg', {'x/a.jpg','y/a.jpg'}, {'a.jpg':['x/a.jpg','y/a.jpg']}) is None


def test_landmark_exact_boundaries_and_outcome_consistency():
    start = pd.Timestamp('2150-01-01')
    frame = pd.DataFrame({'intime':[start]*5, 'outtime':[start+pd.Timedelta(hours=h) for h in [48,47,60,60,60]],
                          'deathtime':[pd.NaT,pd.NaT,start+pd.Timedelta(hours=48),start+pd.Timedelta(hours=49),pd.NaT],
                          'label':[0,0,1,1,1]})
    assert landmark_eligibility(frame).tolist() == [True,False,False,True,False]
