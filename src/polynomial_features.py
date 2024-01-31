import numpy as np


def add_polynomial_features(x: np.ndarray, power: int) -> np.ndarray:
    X = x
    if len(X) < 2:
        return X
    for i in range(2, power + 1):
        X = np.c_[X, x ** i]
    return X
