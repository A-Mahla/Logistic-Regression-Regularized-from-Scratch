import pandas as pd
import numpy as np
import os
import sys
path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(1, path)
from normalization import Minmax
from train import polynomial_train
from plot import plot_f1_score
from best_model import get_best_model, display_best_model


if __name__ == "__main__":

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

    models = polynomial_train(
            classes,
            X,
            Y,
            power=3,
            unique_power=False,
            feature_scaling=Minmax.transform
        )

    best_model = get_best_model(models)
    display_best_model(best_model)
    plot_f1_score(models)
