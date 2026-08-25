from __future__ import division

import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils import column_or_1d, indexable


def _beta_calibration(df, y, sample_weight=None):
    warnings.filterwarnings("ignore")
    df = column_or_1d(df).reshape(-1, 1)
    eps = np.finfo(df.dtype).eps
    df = np.clip(df, eps, 1 - eps)
    y = column_or_1d(y)

    x = np.hstack((df, 1.0 - df))
    x = np.log(x)
    x[:, 1] *= -1

    lr = LogisticRegression(C=99999999999)
    lr.fit(x, y, sample_weight)
    coefs = lr.coef_[0]

    if coefs[0] < 0:
        x = x[:, 1].reshape(-1, 1)
        lr = LogisticRegression(C=99999999999)
        lr.fit(x, y, sample_weight)
        coefs = lr.coef_[0]
        a = 0
        b = coefs[0]
    elif coefs[1] < 0:
        x = x[:, 0].reshape(-1, 1)
        lr = LogisticRegression(C=99999999999)
        lr.fit(x, y, sample_weight)
        coefs = lr.coef_[0]
        a = coefs[0]
        b = 0
    else:
        a = coefs[0]
        b = coefs[1]

    inter = lr.intercept_[0]
    m = minimize_scalar(
        lambda mh: np.abs(b * np.log(1.0 - mh) - a * np.log(mh) - inter),
        bounds=[0, 1],
        method="Bounded",
    ).x
    mapping = [a, b, m]
    return mapping, lr


class _BetaCal(BaseEstimator, RegressorMixin):
    def fit(self, X, y, sample_weight=None):
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)
        self.map_, self.lr_ = _beta_calibration(X, y, sample_weight)
        return self

    def predict(self, S):
        df = column_or_1d(S).reshape(-1, 1)
        eps = np.finfo(df.dtype).eps
        df = np.clip(df, eps, 1 - eps)
        x = np.hstack((df, 1.0 - df))
        x = np.log(x)
        x[:, 1] *= -1

        if self.map_[0] == 0:
            x = x[:, 1].reshape(-1, 1)
        elif self.map_[1] == 0:
            x = x[:, 0].reshape(-1, 1)

        return self.lr_.predict_proba(x)[:, 1]
