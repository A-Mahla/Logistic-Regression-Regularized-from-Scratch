import sys
import os
import dill
from utils import type_checking


@type_checking
def model_save(data: dict, name: str | int, directory: str = 'models') -> None:
    try:
        os.mkdir(
            os.path.join(
                os.path.dirname(__file__),
                '..',
                directory
            )
        )
    except OSError:
        pass
    path = os.path.join(
        os.path.dirname(__file__),
        '..',
        directory,
        f"model_{name}.pkl"
    )
    with open(path, 'wb') as f:
        dill.dump(data, f)


@type_checking
def model_load(name: str | int, directory: str = 'models') -> any:
    path = os.path.join(
            os.path.dirname(__file__),
            '..',
            directory,
            f"model_{name}.pkl"
        )
    try:
        with open(path, 'rb') as f:
            data = dill.load(f)
    except FileNotFoundError:
        print('Please, run \'benchmark_train.py\' before \'selected_model.py\''
              ' to save trained models.')
        sys.exit()
    return data
