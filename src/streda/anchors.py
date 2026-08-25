from __future__ import annotations

import numpy as np
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression

from streda.beta import _BetaCal


def clip_scores(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=float), eps, 1.0 - eps)


class IdentityCalibrator:
    name = "raw"

    def fit(self, scores, targets, sample_weight=None):
        return self

    def predict(self, scores):
        return np.asarray(scores, dtype=float)


class PlattCalibrator:
    name = "platt"
    reference = "Platt (1999/2000), using scikit-learn _SigmoidCalibration"

    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = _SigmoidCalibration()

    def fit(self, scores, targets, sample_weight=None):
        x = np.asarray(scores, dtype=float).reshape(-1)
        y = np.asarray(targets, dtype=int)
        self.model.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, scores):
        x = np.asarray(scores, dtype=float).reshape(-1)
        return self.model.predict(x)


class IsotonicCalibrator:
    name = "isotonic"

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores, targets, sample_weight=None):
        x = np.asarray(scores, dtype=float)
        y = np.asarray(targets, dtype=int)
        self.model.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, scores):
        x = np.asarray(scores, dtype=float)
        return self.model.predict(x)


class BetaCalibrator:
    name = "beta"
    reference = "Kull, Silva Filho, and Flach (2017); adapted from the MIT-licensed betacal Python package"

    def __init__(self, random_state: int = 42, max_iter: int = 1000, eps: float = 1e-6, c: float = 1e12):
        self.random_state = random_state
        self.max_iter = max_iter
        self.eps = eps
        self.c = c
        self.model = _BetaCal()

    def fit(self, scores, targets, sample_weight=None):
        x = np.asarray(scores, dtype=float)
        y = np.asarray(targets, dtype=int)
        self.model.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, scores):
        x = np.asarray(scores, dtype=float)
        return self.model.predict(x)
