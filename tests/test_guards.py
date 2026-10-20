import numpy as np
import pytest
from icu_mm.guards import assert_feature_names, assert_event_window, assert_landmark_cohort


@pytest.mark.parametrize('feature', ['los', 'deathtime', 'hospital_expire_flag', 'text', 'anchor_year_group', 'subject_id'])
def test_forbidden_inputs_rejected(feature):
    with pytest.raises(ValueError):
        assert_feature_names(['heart_rate_mean', feature])


@pytest.mark.parametrize('time', [-.01, 48, 48.01, np.nan, np.inf])
def test_time_boundary_is_half_open(time):
    with pytest.raises(ValueError):
        assert_event_window([0, 47.999, time])


def test_landmark_filters():
    assert_event_window([0, 24, 47.999])
    assert_landmark_cohort([48, 96], [np.nan, 60])
    with pytest.raises(ValueError):
        assert_landmark_cohort([47.99], [np.nan])
    with pytest.raises(ValueError):
        assert_landmark_cohort([96], [48])
