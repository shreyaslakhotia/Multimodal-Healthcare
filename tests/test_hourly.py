import numpy as np
import pandas as pd
import pytest
from icu_mm.data.hourly import aggregate_hourly,summary_features,sum_gcs_components,fahrenheit_to_celsius


def test_hourly_mean_counts_hours_and_first_last_are_observed_only():
    events = pd.DataFrame({'stay_id':[10,10,10],'hour':[2,2,5],'channel':['hr']*3,'value':[60.,80.,100.]})
    values,mask = aggregate_hourly(events,[10,11],['hr'])
    result,names = summary_features(values,mask,['hr'])
    assert values[0,2,0] == 70
    assert result[0,names.index('hr_observed_hours')] == 2
    assert result[0,names.index('hr_first')] == 70
    assert result[0,names.index('hr_last')] == 100
    assert result[0,names.index('hr_mean')] == 85
    assert np.isnan(result[1,:6]).all()
    assert result[1,-1] == 0
    without,names_without = summary_features(values,mask,['hr'],include_counts=False)
    assert without.shape == (2,6)
    assert not any('observed' in name for name in names_without)


def test_gcs_requires_all_three_components_in_the_same_hour():
    events = pd.DataFrame({'stay_id':[10]*5,'hour':[0,0,0,1,1],
                           'component':['eye','verbal','motor','eye','motor'],'value':[4.,5.,6.,4.,6.]})
    result = sum_gcs_components(events)
    assert len(result)==1
    assert result.iloc[0].value==15
    np.testing.assert_allclose(fahrenheit_to_celsius([32,98.6,212]),[0,37,100])


def test_hour_48_and_unknown_patient_are_rejected():
    events = pd.DataFrame({'stay_id':[10],'hour':[48],'channel':['hr'],'value':[60.]})
    with pytest.raises(ValueError):
        aggregate_hourly(events,[10],['hr'])
    events['hour']=0
    with pytest.raises(ValueError):
        aggregate_hourly(events,[99],['hr'])
