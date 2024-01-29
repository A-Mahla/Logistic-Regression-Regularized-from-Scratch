import numpy as np
import os
import dill
import collections.abc as collections


def model_save(data, poly, directory='./models'):
    try:
        os.mkdir(directory)
    except OSError:
        pass
    path = os.path.join(
        os.path.dirname(__file__),
        directory,
        f"model_{poly}.pkl"
    )
    with open(path, 'wb') as f:
        dill.dump(data, f)


def model_load(poly, directory='./models'):
    path = os.path.join(
            os.path.dirname(__file__),
            directory,
            f"model_{poly}.pkl"
        )
    with open(path, 'rb') as f:
        data = dill.load(f)
    return data


def add_polynomial_features(x, power):
    X = x
    if len(X) < 2:
        return X
    for i in range(2, power + 1):
        X = np.c_[X, x ** i]
    return X


def make_poly_predictions(mylr, x, power, feature_scaling):
    return mylr.predict_(add_polynomial_features(feature_scaling(x), power))


def get_iterable(x):
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        return [x]
    return x


def data_spliter(x, y, proportion):

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


def k_fold_cross_validation(x, y, k=5):

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
