from dataclasses import dataclass

import numpy as np


def push(x, stack):
    stack.append(x)


def pop(stack):
    return stack.pop()


def top(stack):
    return stack[-1]


def nextToTop(stack):
    return stack[-2]


def nonleftTurn(a, b, c):
    d1 = b - a
    d2 = c - b
    return np.cross(d1, d2) <= 0


def nonrightTurn(a, b, c):
    d1 = b - a
    d2 = c - b
    return np.cross(d1, d2) >= 0


def slope(a, b):
    ax, ay = a
    bx, by = b
    return (by - ay) / (bx - ax)


def notBelow(t, p1, p2):
    p1x, p1y = p1
    p2x, p2y = p2
    tx, ty = t
    m = (p2y - p1y) / (p2x - p1x)
    b = (p2x * p1y - p1x * p2y) / (p2x - p1x)
    return ty >= tx * m + b


kPrime = None


def algorithm1(P):
    global kPrime
    S = []
    P[-1] = np.array((-1, -1))
    push(P[-1], S)
    push(P[0], S)
    for i in range(1, kPrime + 1):
        while len(S) > 1 and nonleftTurn(nextToTop(S), top(S), P[i]):
            pop(S)
        push(P[i], S)
    return S


def algorithm2(P, S):
    global kPrime
    Sprime = S[::-1]
    F1 = np.zeros((kPrime + 1,))
    for i in range(1, kPrime + 1):
        F1[i] = slope(top(Sprime), nextToTop(Sprime))
        P[i - 1] = P[i - 2] + P[i] - P[i - 1]
        if notBelow(P[i - 1], top(Sprime), nextToTop(Sprime)):
            continue
        pop(Sprime)
        while len(Sprime) > 1 and nonleftTurn(P[i - 1], top(Sprime), nextToTop(Sprime)):
            pop(Sprime)
        push(P[i - 1], Sprime)
    return F1


def algorithm3(P):
    global kPrime
    S = []
    push(P[kPrime + 1], S)
    push(P[kPrime], S)
    for i in range(kPrime - 1, 0 - 1, -1):
        while len(S) > 1 and nonrightTurn(nextToTop(S), top(S), P[i]):
            pop(S)
        push(P[i], S)
    return S


def algorithm4(P, S):
    global kPrime
    Sprime = S[::-1]
    F0 = np.zeros((kPrime + 1,))
    for i in range(kPrime, 1 - 1, -1):
        F0[i] = slope(top(Sprime), nextToTop(Sprime))
        P[i] = P[i - 1] + P[i + 1] - P[i]
        if notBelow(P[i], top(Sprime), nextToTop(Sprime)):
            continue
        pop(Sprime)
        while len(Sprime) > 1 and nonrightTurn(P[i], top(Sprime), nextToTop(Sprime)):
            pop(Sprime)
        push(P[i], Sprime)
    return F0


def prepareData(calibrPoints):
    global kPrime
    ptsSorted = sorted(calibrPoints)
    xs = np.fromiter((p[0] for p in ptsSorted), float)
    ys = np.fromiter((p[1] for p in ptsSorted), float)
    ptsUnique, ptsIndex, ptsInverse, ptsCounts = np.unique(
        xs,
        return_index=True,
        return_counts=True,
        return_inverse=True,
    )
    a = np.zeros(ptsUnique.shape)
    np.add.at(a, ptsInverse, ys)
    w = ptsCounts
    yPrime = a / w
    yCsd = np.cumsum(w * yPrime)
    xPrime = np.cumsum(w)
    kPrime = len(xPrime)
    return yPrime, yCsd, xPrime, ptsUnique


def computeF(xPrime, yCsd):
    global kPrime
    P = {0: np.array((0, 0))}
    P.update({i + 1: np.array((k, v)) for i, (k, v) in enumerate(zip(xPrime, yCsd))})
    S = algorithm1(P)
    F1 = algorithm2(P, S)

    P = {0: np.array((0, 0))}
    P.update({i + 1: np.array((k, v)) for i, (k, v) in enumerate(zip(xPrime, yCsd))})
    P[kPrime + 1] = P[kPrime] + np.array((1.0, 0.0))
    S = algorithm3(P)
    F0 = algorithm4(P, S)
    return F0, F1


def getFVal(F0, F1, ptsUnique, testObjects):
    pos0 = np.searchsorted(ptsUnique, testObjects, side="left")
    pos1 = np.searchsorted(ptsUnique[:-1], testObjects, side="right") + 1
    return F0[pos0], F1[pos1]


def ScoresToMultiProbs(calibrPoints, testObjects):
    yPrime, yCsd, xPrime, ptsUnique = prepareData(calibrPoints)
    F0, F1 = computeF(xPrime, yCsd)
    p0, p1 = getFVal(F0, F1, ptsUnique, testObjects)
    return p0, p1


@dataclass
class VennAbersPrediction:
    lower: np.ndarray
    upper: np.ndarray
    point: np.ndarray


class VennAbersCalibrator:
    name = "venn_abers"
    reference = "Vovk et al. (2015) fast IVAP; MIT-licensed ptocca/VennABERS lineage"

    def __init__(self, eps: float = 1e-6, score_round_decimals: int | None = 6):
        self.eps = eps
        self.score_round_decimals = score_round_decimals
        self.scores_ = None
        self.targets_ = None

    def fit(self, scores, targets, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("Venn-Abers baseline does not support sample weights in this implementation.")
        scores = np.asarray(scores, dtype=float)
        if self.score_round_decimals is not None:
            scores = np.round(scores, self.score_round_decimals)
        self.scores_ = scores
        self.targets_ = np.asarray(targets, dtype=int)
        return self

    def predict_interval(self, scores) -> VennAbersPrediction:
        if self.scores_ is None or self.targets_ is None:
            raise ValueError("Venn-Abers calibrator must be fitted before prediction.")

        raw = np.asarray(scores, dtype=float)
        if self.score_round_decimals is not None:
            raw = np.round(raw, self.score_round_decimals)

        calibration_points = list(zip(self.scores_, self.targets_))
        lower, upper = ScoresToMultiProbs(calibration_points, raw)
        lower = np.clip(lower, self.eps, 1.0 - self.eps)
        upper = np.clip(upper, self.eps, 1.0 - self.eps)
        point = upper / (1.0 - lower + upper)
        point = np.clip(point, self.eps, 1.0 - self.eps)

        return VennAbersPrediction(lower=lower, upper=upper, point=point)

    def predict(self, scores):
        return self.predict_interval(scores).point
