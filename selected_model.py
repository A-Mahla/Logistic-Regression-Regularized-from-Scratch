import pandas as pd
import numpy as np
import sys
import os
path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(1, path)
from plot import plot_f1_score, plot_logistic_model, plot_3D_classifier
from best_model import get_best_model, display_best_model
from save_models import model_load
import matplotlib.pyplot as plt
from polynomial_features import add_polynomial_features
from my_logistic_regression import MyLogisticRegression as MyLR
from utils import type_checking


@type_checking
def multi_predict(model: dict, X: np.ndarray) -> np.ndarray:
    Y_hat = np.array([[]] * X.shape[0])
    for lr_idx, (lr) in enumerate(model['lr']):
        lr_y_hat = lr.predict_proba_(
                add_polynomial_features(
                    model['feature_scaling'](X),
                    best_model['power']
                )
            )
        Y_hat = np.c_[Y_hat, lr_y_hat]
    Y_hat = MyLR.predict_highest_proba_(Y_hat)
    return Y_hat


@type_checking
def upload_models() -> list:
    x_degree = [1, 2, 3]
    lambda_list = [0., 0.2, 0.4, 0.6, 0.8, 1.]
    models = []
    for degree in x_degree:
        for lambda_ in lambda_list:
            models.append(model_load(f'{degree}_lambda-{lambda_}'))
    plot_f1_score(models)
    return models


if __name__ == "__main__":

    # Upload datasets
    datapath = 'datasets/'
    df_features = pd.read_csv(datapath + "solar_system_census.csv")
    df_target = pd.read_csv(datapath + "solar_system_census_planets.csv")
    X = np.array(df_features[["height", "weight", "bone_density"]])
    Y = np.array(df_target[["Origin"]], dtype=int)

    classes = {
        "The flying cities of Venus": 0,
        "United Nations of Earth": 1,
        "Mars Republic": 2,
        "The Asteroids Belt colonies": 3
    }

    # Upload trained models and choose the best one
    models = upload_models()
    best_model = get_best_model(models)
    display_best_model(best_model)

    # Make multiclass prediction
    Y_hat = multi_predict(best_model, X)

    # Plotting results
    plot_3D_classifier(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        Y_hat,
        ["weight", "height", "bone_density"],
        classes.keys(),
        title='Predict Classes',
        num=(
            'model_' + str(best_model['power'])
            + '_lambda_' + str(best_model['lambda_'])
        )
    )

    plot_3D_classifier(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        Y,
        ["weight", "height", "bone_density"],
        classes.keys(),
        title='True Classes',
        num='original targets'
    )

    plt.show()
