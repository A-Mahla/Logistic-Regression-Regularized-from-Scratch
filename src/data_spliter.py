import numpy as np
from utils import type_checking


@type_checking
def data_spliter(
    x: np.ndarray,
    y: np.ndarray,
    proportion: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    def reshape(x):
        return x.reshape((-1, 1)) if x.ndim == 1 else x

    if proportion > 1 or proportion < 0:
        return None
    if proportion == 1 or proportion == 0:
        return (x, x, y, y)
    seed = np.random.SeedSequence()
    x_copy = x.copy()
    y_copy = y.copy()
    np.random.default_rng(seed).shuffle(x_copy)
    np.random.default_rng(seed).shuffle(y_copy)
    x_train, x_test = np.split(x_copy, [int(proportion * len(x_copy))])
    y_train, y_test = np.split(y_copy, [int(proportion * len(y_copy))])
    return reshape(x_train), reshape(x_test), reshape(y_train), reshape(y_test)


@type_checking
def k_fold_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:

    def reshape(x):
        return x.reshape((-1, 1)) if x.ndim == 1 else x

    if k <= 0 or len(x) == 0 or len(y) == 0:
        return None
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    fold_sizes = x.shape[0] // k
    x_test = []
    y_test = []
    x_train = []
    y_train = []
    for i in range(k):
        start, end = i * fold_sizes, (i + 1) * fold_sizes
        x_test.append(reshape(x_shuffled[start:end]))
        y_test.append(reshape(y_shuffled[start:end]))
        x_train.append(
            reshape(np.concatenate([x_shuffled[:start], x_shuffled[end:]]))
        )
        y_train.append(
             reshape(np.concatenate([y_shuffled[:start], y_shuffled[end:]]))
        )
    return x_train, x_test, y_train, y_test
