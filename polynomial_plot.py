import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import get_iterable
from my_logistic_regression import MyLogisticRegression as MyLR
from utils import add_polynomial_features
from matplotlib_config import fig_size


def plot_f1_score(models):

    plt.subplots(figsize=fig_size, num='Polynomial F1_score Evaluation')
    for i, (model) in enumerate(models):
        plt.bar(
            f"$x^{model['power']}$\n$\\lambda={model['lambda_']}$",
            model['f1_score']
        )
    plt.ylabel('F1_score')
    plt.title('Polynomial Metrics Evaluation')
    plt.show()


def colors_and_labels(nb_class, class_labels):
    cmap = 'viridis'
    if nb_class == 2:
        if len(class_labels) == 1:
            class_labels.insert(0, 'Not classified')
        cmap = mcolors.LinearSegmentedColormap.from_list(
                'my_cmap',
                ['darkgrey', 'dodgerblue']
            )
    elif nb_class < 10:
        cmap = mcolors.LinearSegmentedColormap.from_list(
                'my_cmap',
                plt.rcParams['axes.prop_cycle'].by_key()['color'][:nb_class]
            )
    return cmap


def plot_3D_classifier(
    x1, x2, x3, y, feature_labels, class_labels, show=None, title=None
):

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(projection='3d')
    class_labels = get_iterable(class_labels)
    cmap = colors_and_labels(np.unique(y).size, class_labels)

    scatter = ax.scatter(
        x1,
        x2,
        x3,
        c=y,
        cmap=cmap,
    )
    ax.legend(
        title="Planet Origin",
        handles=scatter.legend_elements()[0],
        labels=class_labels
    )
    ax.set_xlabel('\n' + feature_labels[0], linespacing=3.2)
    ax.set_ylabel('\n' + feature_labels[1], linespacing=3.2)
    ax.set_zlabel('\n' + feature_labels[2], linespacing=3.2)
    ax.set_title(title)
    if show is not None:
        plt.show()


def plot_logistic_model(
    model: list,
    x: np.ndarray,
    y: np.ndarray,
    feature_labels: list[str],
    class_labels: list[str],
    target_label: str | None = None
) -> np.ndarray:

    feature_labels = get_iterable(feature_labels)
    class_labels = get_iterable(class_labels)

    y_hat = np.array([[]] * y.shape[0])
    for lr_idx, (lr) in enumerate(model['lr']):
        lr_y_hat = lr.predict_proba_(
                add_polynomial_features(
                    model['feature_scaling'](x), model['power']
                )
            )
        y_hat = np.c_[y_hat, lr_y_hat]
    y_hat = MyLR.predict_highest_proba_(y_hat)

    cmap = colors_and_labels(np.unique(y_hat).size, class_labels)
    for x1_idx in range(x.shape[1]):
        x1 = x[:, x1_idx]
        for x2_idx in range(x1_idx + 1, x.shape[1]):
            x2 = x[:, x2_idx]
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot()
            scatter = ax.scatter(
                x1,
                x2,
                c=y_hat,
                cmap=cmap,
            )
            ax.set_xlabel(feature_labels[x1_idx])
            ax.set_ylabel(feature_labels[x2_idx])
            ax.legend(
                title=target_label,
                handles=scatter.legend_elements()[0],
                labels=class_labels
            )
    plt.show()
    return y_hat
