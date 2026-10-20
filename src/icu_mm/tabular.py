"""Fixed tabular learners fitted solely on the caller's training partition."""
from dataclasses import dataclass
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .guards import assert_feature_names
from .metrics import select_threshold, score


@dataclass
class FoldResult:
    estimator: object
    probabilities: np.ndarray
    threshold: float
    recall80_threshold: float
    recall90_threshold: float
    metrics: dict


def logistic_fold(x_train, y_train, x_validation, y_validation, x_test, y_test,
                  feature_names, include_missingness=True, seed=0):
    assert_feature_names(feature_names)
    if any(x.shape[1] != len(feature_names) for x in (x_train, x_validation, x_test)):
        raise ValueError('Feature names and matrix widths differ')
    pipeline = make_pipeline(
        SimpleImputer(strategy='median', add_indicator=include_missingness, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1., penalty='l2', max_iter=2000, random_state=seed))
    pipeline.fit(x_train, y_train)
    validation_probability = pipeline.predict_proba(x_validation)[:, 1]
    threshold = select_threshold(y_validation, validation_probability)
    recall80 = select_threshold(y_validation, validation_probability, .8)
    recall90 = select_threshold(y_validation, validation_probability, .9)
    # A single probability prediction for the locked outer test set.
    probability = pipeline.predict_proba(x_test)[:, 1]
    metrics = score(y_test, probability, threshold)
    for name, operating_threshold in [('recall80', recall80), ('recall90', recall90)]:
        operating = score(y_test, probability, operating_threshold)
        metrics[f'{name}_precision'] = operating['precision']
        metrics[f'{name}_recall'] = operating['recall']
    return FoldResult(pipeline, probability, threshold, recall80, recall90, metrics)
