import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Callable
from matplotlib_config import fig_size, line_width


class MyLogisticRegression():

    """
    Description:
    My personnal logistic regression to classify things.
    """
    supported_penalities = ['l2']

    def __init__(
        self,
        theta: np.ndarray,
        alpha: float | int = 1e-3,
        max_iter: float | int = 1e3,
        save_loss: bool = False,
        penality: str | None = 'l2',
        lambda_: int | float = 1.0,
        feature_scaling: Callable = lambda x: x
    ):
        error = "MyLogisticRegression.__init__() error args - check prototype"
        if not (
            isinstance(theta, (np.ndarray | list))
            and isinstance(alpha, (float, int))
            and isinstance(lambda_, (float, int))
            and isinstance(max_iter, (float, int))
            and isinstance(save_loss, bool)
            and (isinstance(penality, str) or not penality)
            and callable(feature_scaling)
        ):
            raise TypeError(error)
        self.alpha = alpha
        self.max_iter = int(max_iter)
        self.theta = np.array(theta)
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

    def _add_intercept(self, x: np.ndarray) -> np.ndarray:
        error = ("MyLogisticRegression._add_intercept() arg 1 must be"
                 "a 1 or 2 dimentional np.ndarray")
        if not isinstance(x, np.ndarray) or x.ndim > 2:
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

    def _update_iterate_cost(self, current_loss: float) -> None:
        error = ("MyLogisticRegression._update_iterate_cost() arg 1 must be"
                 "a float")
        if not isinstance(current_loss, float):
            raise TypeError(error)
        if not self.save_loss:
            return
        # save loss() over iteration to plot_convergence.
        if len(self.iterate_cost['iter']) == 0:
            self.iterate_cost['iter'].append(1)
        else:
            self.iterate_cost['iter'].append(self.iterate_cost['iter'][-1] + 1)
        self.iterate_cost['loss'].append(current_loss)
        # ===

    def _l2(self):
        return np.sum(self.theta[1:] ** 2)

    def loss_elem_(
        self, y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15
    ) -> np.ndarray:
        error = ("MyLogisticRegression.loss_elem_() arg 1 and arg 2 must be"
                 "1 or 2 dimentional np.ndarray of same shape")
        if not (
            isinstance(y, np.ndarray)
            and isinstance(y_hat, np.ndarray)
            and y.shape == y_hat.shape
            and y.ndim <= 2
        ):
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

    def loss_(
        self, y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15
    ) -> np.ndarray:
        error = ("MyLogisticRegression.loss_() arg 1 and arg 2 must be"
                 "1 or 2 dimentional np.ndarray of same shape")
        if not (
            isinstance(y, np.ndarray)
            and isinstance(y_hat, np.ndarray)
            and y.shape == y_hat.shape
        ):
            raise TypeError(error)
        # reg = 0
        # if self.penality == 'l2':
        #     reg = self.lambda_ * self._l2() / (2 * y.size)
        loss_elem = np.sum(self.loss_elem_(y, y_hat, eps))
        if loss_elem == 0:
            return 0.0
        return (-1 / y.size) * loss_elem

    def predict_proba_(self, x: np.ndarray) -> np.ndarray:
        error = ("MyLogisticRegression.predict_proba_() arg 1 must be"
                 "a 1 or 2 dimentional np.ndarray")
        if not isinstance(x, np.ndarray) or x.ndim > 2:
            raise TypeError(error)
        X = self._add_intercept(self.feature_scaling(x))
        return self._sigmoid(X @ self.theta)

    def predict_(self,  x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        error = ("MyLogisticRegression.predict_() arg 1 must be"
                 "a 1 or 2 dimentional np.ndarray")
        if not isinstance(x, np.ndarray) or x.ndim > 2:
            raise TypeError(error)
        Y = self.predict_proba_(x)
        Y[Y >= threshold] = 1
        Y[Y < threshold] = 0
        return Y

    @staticmethod
    def predict_class_(
            y: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        error = ("MyLogisticRegression.predict_class_() arg 1 must be"
                 "a 1 or 2 dimentional np.ndarray")
        if not isinstance(y, np.ndarray) or y.ndim > 2:
            raise TypeError(error)
        Y = y.copy()
        Y[Y >= threshold] = 1
        Y[Y < threshold] = 0
        return Y

    @staticmethod
    def predict_highest_proba_(
            y: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        error = ("MyLogisticRegression.predict_highest_proba_() arg 1 must be"
                 "a 1 or 2 dimentional np.ndarray")
        if not isinstance(y, np.ndarray) or y.ndim > 2:
            raise TypeError(error)
        if y.shape[-1] == 1:
            return MyLogisticRegression.predict_class_(y, threshold=threshold)
        else:
            return np.argmax(y, axis=1).reshape((-1, 1))

    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        error = ("MyLogisticRegression.gradient_() arg 1 and arg 2 must be"
                 "a 2 dimentionals np.ndarray of compatible shapes")
        if not (
            isinstance(x, np.ndarray)
            and isinstance(y, np.ndarray)
            and x.ndim == 2 and y.ndim == 2 and y.shape[1] == 1
            and x.shape[0] == y.shape[0] and x.shape[1] == self.theta.shape[0]
        ):
            raise TypeError(error)
        y_hat = self._sigmoid(x @ self.theta)
        self._update_iterate_cost(self.loss_(y, y_hat))
        if self.penality == 'l2':
            theta_ = self.theta.copy()
            theta_.flat[0] = 0
            theta_ *= self.lambda_
            return (x.T @ (y_hat - y) + self.lambda_ * theta_) / y.size
        return (x.T @ (y_hat - y)) / y.size

    def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        error = ("MyLogisticRegression.fit_() arg 1 and arg 2 must be"
                 "a 1 or 2 dimentionals np.ndarray")
        if not (
            isinstance(x, np.ndarray)
            and isinstance(y, np.ndarray)
            and x.ndim <= 2 and y.ndim <= 2
        ):
            raise TypeError(error)
        X = self._add_intercept(self.feature_scaling(x))
        Y = y.reshape((-1, 1)) if y.ndim == 1 else y
        for _ in range(self.max_iter):
            self.theta = self.theta - self.alpha * self.gradient_(X, Y)
        return self.theta

    def plot_convergence(
        self,
    ):
        if len(self.iterate_cost['loss']) == 0:
            print("No loss to plot...")
            return None
        print("loss min value : ", min(self.iterate_cost['loss']))
        fig, ax = plt.subplots(figsize=fig_size)
        plt.xlim((0, self.iterate_cost['iter'][-1]))
        plt.ylim((0, max(self.iterate_cost['loss'])))
        ax.plot(
            self.iterate_cost['iter'],
            self.iterate_cost['loss'],
            linewidth=line_width,
            color='r'
        )
        plt.show()
