import numpy as np
from utils import type_checking


class Minmax():
    def __init__(self):
        self.min = 0.
        self.max = 0.

    @type_checking
    def fit(self, X: np.ndarray) -> 'Minmax':
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    @type_checking
    def apply(self, X: np.ndarray) -> np.ndarray:
        e = 1e-20
        mnmx = (X - self.min) / (self.max - self.min + e)
        return mnmx

    @type_checking
    def unapply(self, X: np.ndarray) -> np.ndarray:
        e = 1e-20
        return (X * (self.max - self.min + e)) + self.min

    @staticmethod
    @type_checking
    def transform(X: np.ndarray) -> np.ndarray:
        e = 1e-20
        mn = X.min(axis=0)
        mx = X.max(axis=0)
        return (X - mn) / (mx - mn + e)


class Zscore():
    def __init__(self):
        self.mean = 0.
        self.std = 0.

    @type_checking
    def fit(self, X: np.ndarray) -> 'Minmax':
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self

    @type_checking
    def apply(self, X: np.ndarray) -> np.ndarray:
        e = 1e-20
        return (X - self.mean) / (self.std + e)

    @type_checking
    def unapply(self, X: np.ndarray) -> np.ndarray:
        e = 1e-20
        return (X * (self.std + e)) + self.mean

    @staticmethod
    @type_checking
    def transform(X: np.ndarray) -> np.ndarray:
        e = 1e-20
        return (X - X.mean) / (X.std + e)
