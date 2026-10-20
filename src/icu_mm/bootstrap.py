"""Patient bootstrap for one locked OOF prediction per patient.

Average seed predictions per patient before calling. Never pool seed copies as
if they were independent subjects. Paired comparisons share resampled indices.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def paired_bootstrap(y, subjects, first, second, iterations=1000, seed=42):
    y, subjects = np.asarray(y), np.asarray(subjects)
    first, second = np.asarray(first, float), np.asarray(second, float)
    if not (y.ndim == first.ndim == second.ndim == subjects.ndim == 1):
        raise ValueError('Expected one-dimensional arrays')
    if not (len(y) == len(subjects) == len(first) == len(second)):
        raise ValueError('Prediction rows must be aligned')
    if len(set(subjects)) != len(y):
        raise ValueError('Expected one prediction per patient after averaging seeds')
    if set(np.unique(y)) != {0, 1} or iterations < 1:
        raise ValueError('Both binary classes and positive iteration count required')
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Predictions must be finite')
    if np.any((first < 0) | (first > 1) | (second < 0) | (second > 1)):
        raise ValueError('Expected probabilities in [0,1]')
    rng = np.random.default_rng(seed)
    metrics = {'auroc': roc_auc_score, 'auprc': average_precision_score}
    distributions = {name: [] for name in metrics}
    skipped = 0
    for _ in range(iterations):
        indices = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[indices])) != 2:
            skipped += 1
            continue
        for name, metric in metrics.items():
            distributions[name].append(metric(y[indices], first[indices]) - metric(y[indices], second[indices]))
    if skipped == iterations:
        raise ValueError('No bootstrap samples contained both outcome classes')
    result = {'requested_draws': iterations, 'valid_draws': iterations - skipped, 'skipped_draws': skipped}
    for name, metric in metrics.items():
        interval = np.percentile(distributions[name], [2.5, 97.5])
        result[name] = {'delta': float(metric(y, first) - metric(y, second)),
                        'lower': float(interval[0]), 'upper': float(interval[1])}
    return result
