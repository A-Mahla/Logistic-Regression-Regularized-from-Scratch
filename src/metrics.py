import numpy as np


def accuracy_score_(y, y_hat):
    if not (
        isinstance(y, np.ndarray)
        and isinstance(y_hat, np.ndarray)
        and y.shape == y_hat.shape
        and len(y_hat) != 0
    ):
        None
    true = y_hat[y_hat == y].size
    return true / y_hat.size


def precision_score_(y, y_hat, pos_label=1):
    if not (
        isinstance(y, np.ndarray)
        and isinstance(y_hat, np.ndarray)
        and y.shape == y_hat.shape
        and len(y_hat) != 0
        and isinstance(pos_label, (int, str))
        and pos_label in y_hat
    ):
        None
    tp = np.sum(np.logical_and(y_hat == pos_label, y == pos_label))
    fp = np.sum(np.logical_and(y_hat == pos_label, y != pos_label))
    if tp + fp == 0:
        return 0.
    return tp / (tp + fp)


def recall_score_(y, y_hat, pos_label=1):
    if not (
        isinstance(y, np.ndarray)
        and isinstance(y_hat, np.ndarray)
        and y.shape == y_hat.shape
        and len(y_hat) != 0
        and isinstance(pos_label, (int, str))
        and pos_label in y_hat
    ):
        None
    tp = np.sum(np.logical_and(y_hat == pos_label, y == pos_label))
    fn = np.sum(np.logical_and(y_hat != pos_label, y == pos_label))
    if tp + fn == 0:
        return 0.
    return tp / (tp + fn)


def f1_score_(y, y_hat, pos_label=1):
    precision = precision_score_(y, y_hat, pos_label=pos_label)
    recall = recall_score_(y, y_hat, pos_label=pos_label)
    if precision + recall == 0:
        return 0.
    return (2 * precision * recall) / (precision + recall)


def f1_score_weighted_(y, y_hat, labels):
    if len(y) == 0:
        return None
    f1_scores = 0.
    for value in labels:
        f1_scores += f1_score_(y, y_hat, pos_label=value) \
                        * np.sum(y == value)
    f1_scores /= y.shape[0]
    return f1_scores
