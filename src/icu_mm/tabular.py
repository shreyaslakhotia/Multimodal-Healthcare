"""Fixed tabular learners fitted solely on the caller's training partition."""
from dataclasses import dataclass
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
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
    return _evaluate(pipeline, x_validation, y_validation, x_test, y_test)


def lightgbm_fold(x_train, y_train, x_validation, y_validation, x_test, y_test,
                  feature_names, include_missingness=True, seed=0, workers=8):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    assert_feature_names(feature_names)
    if any(x.shape[1] != len(feature_names) for x in (x_train,x_validation,x_test)):
        raise ValueError('Feature names and matrix widths differ')
    imputer = SimpleImputer(strategy='median', add_indicator=include_missingness, keep_empty_features=True)
    training = imputer.fit_transform(x_train)
    validation = imputer.transform(x_validation)
    model = LGBMClassifier(objective='binary', metric='auc', learning_rate=.05, num_leaves=15,
                           min_child_samples=50, n_estimators=1000, reg_lambda=1.,
                           n_jobs=workers, random_state=seed, deterministic=True,
                           force_col_wise=True, verbosity=-1)
    model.fit(training, y_train, eval_set=[(validation,y_validation)], eval_metric='auc',
              callbacks=[early_stopping(50, first_metric_only=True, verbose=False), log_evaluation(0)])
    pipeline = Pipeline([('simpleimputer',imputer),('lightgbm',model)])
    return _evaluate(pipeline, x_validation, y_validation, x_test, y_test)


def _evaluate(pipeline, x_validation, y_validation, x_test, y_test):
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
