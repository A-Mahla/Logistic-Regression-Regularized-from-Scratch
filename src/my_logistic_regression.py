import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from matplotlib_config import fig_size, line_width
from utils import type_checking
from data_spliter import mini_batches


class MyLogisticRegression():

    """
    Description:
    My personnal logistic regression to classify things.
    """

    supported_penalities = ['l2']
    supported_gradient = ['batch', 'mini_batch', 'stochastic']

    @type_checking
    def __init__(
        self,
        theta: np.ndarray | list,
        alpha: float | int = 1e-3,
        max_iter: float | int = 1e3,
        save_loss: bool = False,
        penality: str | None = 'l2',
        lambda_: int | float = 0.0,
        feature_scaling: Callable = lambda x: x
    ):

        self.alpha = alpha
        self.max_iter = int(max_iter)
        self.theta = np.array(theta, dtype=float)
        if self.theta.ndim == 1:
            self.theta = self.theta.reshape((-1, 1))
        elif self.theta.ndim > 2:
            raise TypeError("MyLogisticRegression.__init__(): 'theta' arg"
                            "must be 1 or 2 dimentional array")
        self.lambda_ = lambda_ if penality in self.supported_penalities else 0.
        self.penality = penality
        self.epsilon = 0.
        self.iterate_cost = {
            'iter': [],
            'loss': []
        }
        self.save_loss = save_loss
        self.feature_scaling = feature_scaling

    @type_checking
    def _add_intercept(self, x: np.ndarray) -> np.ndarray:
        error = ("MyLogisticRegression._add_intercept() arg 1 must be"
                 "a 1 or 2 dimentional np.ndarray")
        if x.ndim > 2:
            raise TypeError(error)
        if x.ndim == 1:
            return np.c_[np.ones((x.shape[0], 1)), x.T]
        return np.c_[np.ones((x.shape[0], 1)), x]

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        error = ("MyLogisticRegression._sigmoid() arg 1 must be"
                 "a 1 or 2 dimentional np.ndarray")
        if not isinstance(x, np.ndarray) or x.ndim > 2:
            raise TypeError(error)
        return 1 / (1 + np.exp(-x))

    @type_checking
    def _update_iterate_cost(
        self,
        current_loss: float
    ) -> None:
        # save loss() over iteration to plot_convergence.
        if len(self.iterate_cost['iter']) == 0:
            self.iterate_cost['iter'].append(1)
        else:
            self.iterate_cost['iter'].append(self.iterate_cost['iter'][-1] + 1)
        self.iterate_cost['loss'].append(current_loss)
        # ===

    def _l2(self):
        return np.sum(self.theta[1:] ** 2)

    @type_checking
    def loss_elem_(
        self, y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15
    ) -> np.ndarray:
        error = ("MyLogisticRegression.loss_elem_() arg 1 and arg 2 must be"
                 "1 or 2 dimentional np.ndarray of same shape")
        if not (y.shape == y_hat.shape and y.ndim <= 2):
            raise TypeError(error)
        Y = y
        if eps == 0. and y_hat.all() == y.all():
            return 0.0
        elif eps == 0. and (np.any(y_hat == 0) or np.any(y_hat == 1)):
            return float('-inf')
        Y_hat = y_hat.astype(float)
        Y_hat[Y_hat < eps] = eps
        Y_hat[Y_hat > 1 - eps] = 1 - eps
        one_mtx = np.ones(Y.shape)
        return Y * np.log(Y_hat) + (one_mtx - Y) * np.log(one_mtx - Y_hat)

    @type_checking
    def loss_(
        self, y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15
    ) -> np.ndarray:
        error = ("MyLogisticRegression.loss_() arg 1 and arg 2 must be"
                 "1 or 2 dimentional np.ndarray of same shape")
        if y.shape != y_hat.shape:
            raise TypeError(error)
        # reg = 0
        # if self.penality == 'l2':
        #     reg = self.lambda_ * self._l2() / (2 * y.size)
        loss_elem = np.sum(self.loss_elem_(y, y_hat, eps))
        if loss_elem == 0:
            return 0.0
        return (-1 / y.size) * loss_elem

    @type_checking
    def predict_proba_(self, x: np.ndarray) -> np.ndarray:
        error = ("MyLogisticRegression.predict_proba_() arg 1 must be"
                 "a 1 or 2 dimentional np.ndarray")
        if x.ndim > 2:
            raise TypeError(error)
        X = self._add_intercept(self.feature_scaling(x))
        return self._sigmoid(X @ self.theta)

    @type_checking
    def predict_(self,  x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        error = ("MyLogisticRegression.predict_() arg 1 must be"
                 "a 1 or 2 dimentional np.ndarray")
        if x.ndim > 2:
            raise TypeError(error)
        Y = self.predict_proba_(x)
        Y[Y >= threshold] = 1
        Y[Y < threshold] = 0
        return Y

    @staticmethod
    @type_checking
    def predict_class_(
            y: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        error = ("MyLogisticRegression.predict_class_() arg 1 must be"
                 "a 1 or 2 dimentional np.ndarray")
        if y.ndim > 2:
            raise TypeError(error)
        Y = y.copy()
        Y[Y >= threshold] = 1
        Y[Y < threshold] = 0
        return Y

    @staticmethod
    @type_checking
    def predict_highest_proba_(
            y: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        error = ("MyLogisticRegression.predict_highest_proba_() arg 1 must be"
                 "a 1 or 2 dimentional np.ndarray")
        if y.ndim > 2:
            raise TypeError(error)
        if y.shape[-1] == 1:
            return MyLogisticRegression.predict_class_(y, threshold=threshold)
        else:
            return np.argmax(y, axis=1).reshape((-1, 1))

    def gradient_(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:

        error = ("MyLogisticRegression.gradient_() arg 1 and arg 2 must be"
                 "a 2 dimentionals non-empty np.ndarray of compatible shapes")
        if not (
            X.ndim == 2 and Y.ndim == 2 and Y.shape[1] == 1 and len(X) != 0
            and X.shape[0] == Y.shape[0] and X.shape[1] == self.theta.shape[0]
        ):
            raise TypeError(error)
        Y_hat = self._sigmoid(X @ self.theta)
        if self.save_loss:
            self._update_iterate_cost(self.loss_(Y, Y_hat))
        if self.penality == 'l2':
            theta_ = self.theta.copy()
            theta_.flat[0] = 0
            theta_ *= self.lambda_
            return (X.T @ (Y_hat - Y) + self.lambda_ * theta_) / Y.size
        return (X.T @ (Y_hat - Y)) / Y.size

    @type_checking
    def fit_(
        self,
        x: np.ndarray,
        y: np.ndarray,
        gradient_type: str = 'batch'
    ) -> np.ndarray:

        error = ("MyLogisticRegression.fit_() arg 1 and arg 2 must be"
                 "a 1 or 2 dimentionals np.ndarray")
        if not (x.ndim <= 2 and y.ndim <= 2):
            raise TypeError(error)
        X = self._add_intercept(self.feature_scaling(x))
        Y = y.reshape((-1, 1)) if y.ndim == 1 else y
        i = 0
        datas = [(X, Y)]
        if gradient_type == 'mini_batch':
            datas = mini_batches(X, Y, 32)
        elif gradient_type == 'stochastic':
            datas = mini_batches(X, Y, 1)
        for _ in range(self.max_iter):
            for (sub_x, sub_y) in datas:
                if i >= self.max_iter:
                    break
                gradient = self.gradient_(sub_x, sub_y)
                self.theta = self.theta - self.alpha * gradient
                i += 1
            if i >= self.max_iter:
                break
        return self.theta

    def plot_convergence(self, num='Loss vs iterations'):
        if len(self.iterate_cost['loss']) == 0:
            print("No loss to plot...")
            return None
        print("loss min value : ", min(self.iterate_cost['loss']))
        fig, ax = plt.subplots(figsize=fig_size, num=num)
        plt.xlim((0, self.iterate_cost['iter'][-1]))
        plt.ylim(
            (min(self.iterate_cost['loss']), max(self.iterate_cost['loss']))
        )
        ax.plot(
            self.iterate_cost['iter'],
            self.iterate_cost['loss'],
            linewidth=line_width,
            color='r'
        )
        plt.show()
