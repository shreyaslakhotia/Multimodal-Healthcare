import numpy as np
from icu_mm.tabular import logistic_fold


def test_heldout_perturbation_does_not_refit_pipeline():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(160, 3))
    y = (x[:, 0] + rng.normal(size=160) > 0).astype(int)
    x[::7, 1] = np.nan
    first = logistic_fold(x[:100], y[:100], x[100:130], y[100:130], x[130:], y[130:], ['heart_rate_mean', 'glucose_mean', 'age'])
    second = logistic_fold(x[:100], y[:100], x[100:130] * 1000, y[100:130], x[130:] * 1000, y[130:], ['heart_rate_mean', 'glucose_mean', 'age'])
    for step, attribute in [('simpleimputer', 'statistics_'), ('standardscaler', 'mean_'), ('logisticregression', 'coef_')]:
        np.testing.assert_array_equal(getattr(first.estimator.named_steps[step], attribute), getattr(second.estimator.named_steps[step], attribute))
