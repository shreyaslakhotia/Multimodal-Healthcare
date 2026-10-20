"""Persist caller-provided patient partitions; never split within a patient."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


def make_partitions(y, subjects, seed=42):
    y, subjects = np.asarray(y), np.asarray(subjects)
    if len(y) != len(subjects) or len(y) == 0:
        raise ValueError("Labels and subjects must have equal nonzero lengths")
    # The primary cohort has exactly one ICU stay per patient.
    if len(np.unique(subjects)) != len(subjects):
        raise ValueError("Primary cohort must have exactly one row per patient")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary outcome classes required")
    outer = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(y)), y, subjects)):
        outer[test] = fold
    parts = []
    for fold in range(5):
        training = np.flatnonzero(outer != fold)
        test = np.flatnonzero(outer == fold)
        inner = StratifiedShuffleSplit(1, test_size=0.15, random_state=seed + fold)
        fit_idx, val_idx = next(inner.split(training, y[training]))
        train, val = training[fit_idx], training[val_idx]
        assert len(set(subjects[train]) & set(subjects[val])) == 0
        assert len(set(subjects[train]) & set(subjects[test])) == 0
        assert len(set(subjects[val]) & set(subjects[test])) == 0
        parts.append({"train": train, "validation": val, "test": test})
    return outer, parts
