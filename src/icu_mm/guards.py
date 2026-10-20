"""Fail closed on forbidden inputs and event-window violations."""
import numpy as np

FORBIDDEN = frozenset({
    'dod', 'deathtime', 'dischtime', 'outtime', 'los', 'last_careunit',
    'discharge_location', 'hospital_expire_flag', 'drgcodes', 'diagnoses_icd',
    'race', 'insurance', 'language', 'marital_status', 'admittime', 'edregtime',
    'edouttime', 'intime', 'charttime', 'storetime', 'text', 'text_augment',
    'subject_id', 'stay_id', 'hadm_id', 'fold', 'label',
})


def assert_feature_names(names):
    """Metadata may exist in cohort tables, never in actual model matrices."""
    invalid = [n for n in names if n.lower() in FORBIDDEN or n.lower().startswith('anchor_year')]
    if invalid:
        # Names are schema, not values; never include rows in exceptions.
        raise ValueError(f'Forbidden model feature names: {invalid}')


def assert_event_window(relative_hours):
    times = np.asarray(relative_hours, dtype=float)
    if not np.isfinite(times).all() or np.any((times < 0) | (times >= 48)):
        raise ValueError('Event outside the half-open [0,48) hour window')


def assert_landmark_cohort(los_hours, death_relative_hours):
    los = np.asarray(los_hours, dtype=float)
    death = np.asarray(death_relative_hours, dtype=float)
    if los.shape != death.shape:
        raise ValueError('Cohort array shapes differ')
    if not np.isfinite(los).all() or np.any(los < 48):
        raise ValueError('ICU length of stay below landmark or missing')
    # NaN means no recorded death timestamp; +/- infinity is invalid.
    if np.isinf(death).any() or np.any(np.isfinite(death) & (death <= 48)):
        raise ValueError('Death recorded at or before landmark, or invalid death time')
