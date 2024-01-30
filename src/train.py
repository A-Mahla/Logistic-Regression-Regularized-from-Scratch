from my_logistic_regression import MyLogisticRegression as MyLR
from data_spliter import data_spliter, k_fold_cross_validation
from save_models import model_save
from polynomial_features import add_polynomial_features
from metrics import f1_score_, f1_score_weighted_, accuracy_score_
from normalization import Minmax
import numpy as np
from typing import Callable


def logistic_model_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    theta: np.ndarray,
    class_label: str,
    class_value: int | str,
    alpha=1e-2,
    max_iter=1e5,
    lambda_=0
) -> tuple[MyLR, np.ndarray, float]:

    def get_classed_y(y, class_value):
        y_copy = y.copy()
        y_copy[y_copy == class_value] = -1
        y_copy[y_copy != -1] = 0
        y_copy[y_copy == -1] = 1
        return y_copy

    y_train_copy = get_classed_y(y_train, class_value)
    y_test_copy = get_classed_y(y_test, class_value)

    penality = 'l2' if lambda_ != 0. else None
    mylr = MyLR(
        theta,
        alpha=alpha,
        max_iter=max_iter,
        penality=penality,
        lambda_=lambda_
    )
    mylr.fit_(x_train, y_train_copy)
    y_hat = mylr.predict_proba_(x_test)
    f1 = f1_score_(y_test_copy, MyLR.predict_class_(y_hat))
    return mylr, y_hat, f1


def logistic_classifier_train(
    classes: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    theta: np.ndarray,
    alpha: float | int = 1e-1,
    max_iter: float | int = 1e5,
    lambda_: float | int = 0,
) -> tuple[list[np.ndarray], float, float] | None:

    if not isinstance(classes, dict):
        return None
    lrs = []
    y_hats = np.array([[]] * y_test.shape[0])

    for key, value in classes.items():
        lr, lr_y_hat, _ = logistic_model_train(
                x_train,
                y_train,
                x_test,
                y_test,
                theta,
                key,
                value,
                alpha=alpha,
                max_iter=max_iter,
                lambda_=lambda_,
            )
        lrs.append(lr)
        y_hats = np.c_[y_hats, lr_y_hat]
    y_hats = MyLR.predict_highest_proba_(y_hats)
    if len(classes) > 1:
        f1_score_weighted = f1_score_weighted_(y_test, y_hats, classes.values())
    accuracy = accuracy_score_(y_test, y_hats)
    return lrs, f1_score_weighted, accuracy


def one_poly_train(
    classes: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    theta: np.ndarray,
    alpha: float | int = 1e-2,
    max_iter: float | int = 1e5,
    power: int = 1,
    lambda_: float | int = 0,
) -> tuple[list[np.ndarray], float, float]:

    lrs = []
    f1_scores = []
    accuracies = []
    for X_train, X_test, Y_train, Y_test in zip(
        x_train, x_test, y_train, y_test
    ):
        X_poly_train = add_polynomial_features(X_train, power)
        X_poly_test = add_polynomial_features(X_test, power)
        lrs_class, f1_score_weighted, accuracy = logistic_classifier_train(
            classes,
            X_poly_train,
            Y_train,
            X_poly_test,
            Y_test,
            theta,
            alpha=alpha,
            max_iter=max_iter,
            lambda_=lambda_,
        )
        lrs.append(lrs_class)
        f1_scores.append(f1_score_weighted)
        accuracies.append(accuracy)
    mean_f1 = np.mean(f1_scores)
    mean_accuracy = np.mean(accuracies)
    print(f'\t    Multi-Class Logistic Model '
          f'(lambda = {lambda_:3}):'
          ' accuracy:'
          f' \033[38;5;109m{mean_accuracy * 100:5.5}%\033[0m,'
          ' weighted f1 score:'
          f' \033[38;5;109m{mean_f1 * 100:5.5}%\033[0m')
    return lrs[np.argmax(f1_scores)], mean_f1, mean_accuracy


def polynomial_train(
    classes: dict,
    X: np.ndarray,
    Y: np.ndarray,
    power: int = 1,
    unique_power: bool = True,
    feature_scaling: Callable = Minmax.transform,
    proportion: float = None,
    save_model: bool = True
) -> list[dict]:

    if power > 6 or power < 1:
        return None
    range_power = power if not unique_power else 1
    lambdas_ = [0., 0.2, 0.4, 0.6, 0.8, 1.]
    thetas = [[0.]] * (X.shape[1] + 1)
    config = {
        'max_iter': [1e5, 1e5, 1e5, 1e5, 1e6, 1e6],
        'alpha': [1, 1, 1, 1e-2, 1e-8, 1e-9],
        'thetas': [
            thetas,
            thetas + [[0.]] * X.shape[1] * 1,
            thetas + [[0.]] * X.shape[1] * 2,
            thetas + [[0.]] * X.shape[1] * 3,
            thetas + [[0.]] * X.shape[1] * 4,
            thetas + [[0.]] * X.shape[1] * 5,
        ]
    }
    lrs = []

    X_scale = feature_scaling(X)
    if not proportion:
        X_train, X_test, Y_train, Y_test = k_fold_cross_validation(X_scale, Y)
    else:
        X_train, X_test, Y_train, Y_test = data_spliter(X_scale, Y, proportion)
        X_train, X_test, Y_train, Y_test = [X_train], [X_test], [Y_train], [Y_test]
    print(
        '\n ===   Polynomial Training - '
        f'Data proportion train/test: {len(X_train[0])}/{len(X_test[0])} '
        f'({len(X_train[0]) / len(X) * 100}%/'
        f'{len(X_test[0]) / len(X) * 100}%)', end=''
    )
    if not proportion:
        print(' - K-fold Cross Validation set activate (k = 5)', end='')
    print('\n')
    for i in range(range_power):
        target_power = power if unique_power else i + 1
        print(f'\n\t\033[38;5;109mPolynomial Degree {target_power}\033[0m :\n')
        for lambda_ in lambdas_:
            lrs_class, f1_score, accuracy = one_poly_train(
                    classes,
                    X_train,
                    Y_train,
                    X_test,
                    Y_test,
                    config['thetas'][target_power - 1],
                    alpha=config['alpha'][target_power - 1],
                    max_iter=config['max_iter'][target_power - 1],
                    power=target_power,
                    lambda_=lambda_
                )
            lrs.append({
                'lr': lrs_class,
                'feature_scaling': feature_scaling,
                'lambda_': lambda_,
                'power': target_power,
                'f1_score': f1_score,
                'accuracy': accuracy
            })
            if save_model:
                model_save(lrs[-1], f'{target_power}_lambda-{lambda_}')
        print()
    return lrs
