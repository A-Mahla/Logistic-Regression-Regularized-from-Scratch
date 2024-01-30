import sys
import os
import dill


def model_save(data, poly, directory='models'):
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
        f"model_{poly}.pkl"
    )
    with open(path, 'wb') as f:
        dill.dump(data, f)


def model_load(poly, directory='models'):
    path = os.path.join(
            os.path.dirname(__file__),
            '..',
            directory,
            f"model_{poly}.pkl"
        )
    try:
        with open(path, 'rb') as f:
            data = dill.load(f)
    except FileNotFoundError:
        print('Please, run \'benchmark_train.py\' before \'selected_model.py\''
              ' to save trained models.')
        sys.exit()
    return data
