import numpy as np
from utils import type_checking


@type_checking
def accuracy_score_(y: np.ndarray, y_hat: np.ndarray) -> float:
    if not (
        y.shape == y_hat.shape
        and len(y_hat) != 0
    ):
        None
    true = y_hat[y_hat == y].size
    return true / y_hat.size


@type_checking
def precision_score_(
    y: np.ndarray,
    y_hat: np.ndarray,
    pos_label: np.number | int | str = 1
) -> float:

    if not (
        y.shape == y_hat.shape
        and len(y_hat) != 0
        and pos_label in y_hat
    ):
        None
    tp = np.sum(np.logical_and(y_hat == pos_label, y == pos_label))
    fp = np.sum(np.logical_and(y_hat == pos_label, y != pos_label))
    if tp + fp == 0:
        return 0.
    return tp / (tp + fp)


@type_checking
def recall_score_(
    y: np.ndarray,
    y_hat: np.ndarray,
    pos_label: np.number | int | str = 1
) -> float:

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


@type_checking
def f1_score_(
    y: np.ndarray,
    y_hat: np.ndarray,
    pos_label: np.number | int | str = 1
) -> float:

    precision = precision_score_(y, y_hat, pos_label=pos_label)
    recall = recall_score_(y, y_hat, pos_label=pos_label)
    if precision + recall == 0:
        return 0.
    return (2 * precision * recall) / (precision + recall)


@type_checking
def f1_score_weighted_(
    y: np.ndarray,
    y_hat: np.ndarray,
) -> float:

    if len(y) == 0:
        return None
    f1_scores = 0.
    labels = np.unique(y)
    for value in labels:
        f1_scores += f1_score_(y, y_hat, pos_label=value) \
                        * np.sum(y == value)
    f1_scores /= y.shape[0]
    return f1_scores
