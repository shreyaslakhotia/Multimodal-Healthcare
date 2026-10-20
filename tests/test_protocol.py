import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from icu_mm.splits import make_partitions
from icu_mm.preprocessing import HourlyNormalizer
from icu_mm.metrics import score, select_threshold


def test_patient_partition_determinism_and_isolation():
    subjects = np.arange(1000)
    labels = (subjects % 5 == 0).astype(int)
    folds, portions = make_partitions(labels, subjects)
    other_folds, other = make_partitions(labels, subjects)
    np.testing.assert_array_equal(folds, other_folds)
    for p, q in zip(portions, other):
        for key in p:
            np.testing.assert_array_equal(p[key], q[key])
        sets = [set(subjects[p[key]]) for key in ('train', 'validation', 'test')]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert len(set.union(*sets)) == 1000


def test_duplicate_patient_rejected():
    with pytest.raises(ValueError, match='one row per patient'):
        make_partitions([0, 1], [10, 10])


def test_heldout_perturbation_cannot_change_training_statistics():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(20, 48, 3))
    observed = np.ones(train.shape, dtype=bool)
    normalizer = HourlyNormalizer().fit(train, observed)
    original_mean, original_scale = normalizer.mean_.copy(), normalizer.scale_.copy()
    heldout = rng.normal(size=(4, 48, 3))
    normalizer.transform(heldout, np.ones(heldout.shape, dtype=bool))
    normalizer.transform(heldout * 1e10, np.ones(heldout.shape, dtype=bool))
    np.testing.assert_array_equal(normalizer.mean_, original_mean)
    np.testing.assert_array_equal(normalizer.scale_, original_scale)


def test_imputation_does_not_look_forward():
    normalizer = HourlyNormalizer().fit(np.arange(48.)[None, :, None], np.ones((1, 48, 1), bool))
    x, mask = np.zeros((1, 48, 1)), np.zeros((1, 48, 1), bool)
    x[:, 3, :] = 40
    mask[:, 3, :] = True
    result = normalizer.transform(x, mask)
    np.testing.assert_array_equal(result[:, :3], 0)
    expected = np.full_like(result[:, 3:], ((40 - normalizer.mean_) / normalizer.scale_)[0])
    np.testing.assert_allclose(result[:, 3:], expected, rtol=1e-6)


def test_metrics_match_sklearn_and_validation_threshold_is_optimal():
    y = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    p = np.array([.1, .8, .6, .7, .05, .4, .3, .2])
    threshold = select_threshold(y, p)
    result = score(y, p, threshold)
    prediction = p >= threshold
    assert result['auroc'] == roc_auc_score(y, p)
    assert result['auprc'] == average_precision_score(y, p)
    assert result['precision'] == precision_score(y, prediction)
    assert result['recall'] == recall_score(y, prediction)
    assert result['f1'] == f1_score(y, prediction)
    assert sum(result[k] for k in ('tn', 'fp', 'fn', 'tp')) == len(y)
    assert result['f1'] == max(f1_score(y, p >= t) for t in np.unique(p))
    target = select_threshold(y, p, recall_target=.8)
    assert recall_score(y, p >= target) >= .8
