

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
