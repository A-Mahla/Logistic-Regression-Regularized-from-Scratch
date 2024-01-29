import pandas as pd
import numpy as np
from normalization import Minmax
from polynomial_train import polynomial_train
from polynomial_plot import plot_f1_score


def get_best_model(models):
    best_model = models[0]
    for model in models:
        if model['f1_score'] > best_model['f1_score']:
            best_model = model
    return best_model


def display_best_model(model):
    print(
        '\033[93m'
        '\n\t==========================================================\n\n'
        '\033[00m'
        '\t    The best Polynomial Regularized Logistic Regression Model is:\n'
        f"\t     - power: \033[38;5;109m{model['power']}\033[0m\n"
        f"\t     - lambda: \033[38;5;109m{model['lambda_']}\033[0m\n"
        f"\t     - f1_score: \033[38;5;109m{model['f1_score'] * 100}%\033[0m"
        '\033[93m'
        '\n\n\t==========================================================\n'
        '\033[00m'
    )


if __name__ == "__main__":

    df_features = pd.read_csv("solar_system_census.csv")
    df_target = pd.read_csv("solar_system_census_planets.csv")
    X = np.array(df_features[["height", "weight", "bone_density"]])
    Y = np.array(df_target[["Origin"]], dtype=int)

    classes = {
        "The flying cities of Venus": 0,
        "United Nations of Earth": 1,
        "Mars Republic": 2,
        "The Asteroids Belt colonies": 3
    }

    models = polynomial_train(
            classes,
            X,
            Y,
            power=3,
            unique_power=False,
            feature_scaling=Minmax.lambda_apply,
        )

    best_model = get_best_model(models)
    display_best_model(best_model)
    plot_f1_score(models)
