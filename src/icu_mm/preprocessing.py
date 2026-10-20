"""Train-only event-value statistics and within-window forward fill."""
import numpy as np


class HourlyNormalizer:
    def fit(self, values, observed):
        values = np.asarray(values, dtype=float)
        observed = np.asarray(observed, dtype=bool)
        self._validate(values, observed)
        finite = observed & np.isfinite(values)
        counts = finite.sum(axis=(0, 1))
        if np.any(counts == 0):
            raise ValueError("A channel has no valid training observations")
        self.mean_ = np.where(finite, values, 0).sum(axis=(0, 1)) / counts
        variance = np.where(finite, (values - self.mean_) ** 2, 0).sum(axis=(0, 1)) / counts
        self.scale_ = np.sqrt(variance)
        self.scale_[self.scale_ == 0] = 1
        return self

    @staticmethod
    def _validate(values, observed):
        if values.ndim != 3 or values.shape[1] != 48 or values.shape != observed.shape:
            raise ValueError("Expected matching [patients, 48, channels] arrays")

    def transform(self, values, observed):
        values = np.asarray(values, dtype=float)
        observed = np.asarray(observed, dtype=bool)
        self._validate(values, observed)
        if values.shape[2] != len(self.mean_):
            raise ValueError("Channel count changed")
        output = np.empty_like(values)
        previous = np.broadcast_to(self.mean_, (values.shape[0], values.shape[2])).copy()
        for hour in range(48):
            valid = observed[:, hour] & np.isfinite(values[:, hour])
            previous = np.where(valid, values[:, hour], previous)
            output[:, hour] = previous
        return ((output - self.mean_) / self.scale_).astype(np.float32)
