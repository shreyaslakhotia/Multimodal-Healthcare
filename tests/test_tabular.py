import numpy as np
from icu_mm.tabular import logistic_fold, lightgbm_fold


def test_heldout_perturbation_does_not_refit_pipeline():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(160, 3))
    y = (x[:, 0] + rng.normal(size=160) > 0).astype(int)
    x[::7, 1] = np.nan
    first = logistic_fold(x[:100], y[:100], x[100:130], y[100:130], x[130:], y[130:], ['heart_rate_mean', 'glucose_mean', 'age'])
    second = logistic_fold(x[:100], y[:100], x[100:130] * 1000, y[100:130], x[130:] * 1000, y[130:], ['heart_rate_mean', 'glucose_mean', 'age'])
    for step, attribute in [('simpleimputer', 'statistics_'), ('standardscaler', 'mean_'), ('logisticregression', 'coef_')]:
        np.testing.assert_array_equal(getattr(first.estimator.named_steps[step], attribute), getattr(second.estimator.named_steps[step], attribute))


def test_lightgbm_imputer_is_training_only_and_test_scores_are_finite():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(500,3))
    y = (x[:,0] + rng.normal(size=500) > 0).astype(int)
    x[::11,1] = np.nan
    result = lightgbm_fold(x[:300],y[:300],x[300:400]*100,y[300:400],x[400:],y[400:],['heart_rate_mean','glucose_mean','age'],workers=1)
    np.testing.assert_allclose(result.estimator.named_steps['simpleimputer'].statistics_,np.nanmedian(x[:300],axis=0))
    assert np.isfinite(result.probabilities).all()
    assert result.metrics['n'] == 100
