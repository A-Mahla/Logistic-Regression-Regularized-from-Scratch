
# Logistic Regression Regularized from Scratch

## Overview
This project provides a comprehensive implementation of Regularized Logistic Regression entirely from scratch, utilizing only `numpy` for linear algebra operations and `pandas` for dataset manipulation. Designed for educational purposes and real-world application, the repository offers a deep dive into the fundamentals of logistic regression, feature scaling, metrics evaluation, and data splitting techniques without relying on high-level machine learning libraries.

## Features
- **MyLogisticRegression Class**: A complete implementation capable of performing regularized logistic regression with various gradient descent techniques including Gradient Descent (GD), Mini-batch Gradient Descent (MGD), and Stochastic Gradient Descent (SGD).
- **Feature Scaling**: Includes custom classes for Min-max scaling and Z-score normalization to prepare your data for optimal regression performance.
- **Logistic Metrics**: Compute essential metrics such as accuracy, precision, recall, and F1 score to evaluate model performance directly from the predictions.
- **Polynomial Feature Addition**: Enhance your model with polynomial features to capture more complex relationships in the data.
- **Data Splitting Tools**: Leverage utilities like K-fold cross-validation set generation and batch splitting for more effective training and validation processes.

## Getting Started

### Prerequisites
Ensure you have the following installed (see requirements.txt):
- Python 3.x
- numpy
- pandas
- matplolib
- dill

### Installation
Clone the repository to your local machine:
```
git clone https://github.com/yourusername/Logistic-Regression-Regularized-from-Scratch.git
```

### Usage
The project includes two main scripts for demonstration and testing purposes:

1. **benchmark_train.py**: Run this script to perform a benchmark across different hyperparameters, choosing the best performing model and saving each model's parameters for further analysis.

   ```
   python benchmark_train.py
   ```

2. **selected_model.py**: Use this script to load the best trained model and visualize its performance with a plot of the decision boundary or other relevant metrics.

   ```
   python selected_model.py
   ```

## Contributing
We welcome contributions and suggestions! Feel free to fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Special thanks to the open-source community for the inspiration and resources to make this project possible.
- Anyone who contributes to improving this project through issues, pull requests, or feedback.

## Contact
For any queries or feedback, please open an issue in the GitHub issue tracker.
