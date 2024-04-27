import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.metrics import classification_report, confusion_matrix


def fit_and_predict(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    return (classification_report(y_test, y_pred, output_dict=True),
            classification_report(y_test, y_pred),
            confusion_matrix(y_test, y_pred).ravel())


def plot_confusion_matrix(matrix, cmap):
    tp, fp, fn, tn = matrix

    plt.figure(figsize=(8, 8))
    plt.imshow([[tp, fp], [fn, tn]], interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()

    plt.xticks([0, 1], ['Predicted Positive', 'Predicted Negative'])
    plt.yticks([0, 1], ['Actual Positive', 'Actual Negative'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.text(0, 0, tp, ha='center', va='center', color='white')
    plt.text(1, 0, fp, ha='center', va='center', color='black')
    plt.text(0, 1, fn, ha='center', va='center', color='black')
    plt.text(1, 1, tn, ha='center', va='center', color='white')

    plt.show()


def print_avg_metrics(full_metrics):
    accuracy = np.mean([metric['accuracy'] for metric in full_metrics])
    precision = np.mean([metric['macro avg']['precision'] for metric in full_metrics])
    recall = np.mean([metric['macro avg']['recall'] for metric in full_metrics])
    f1_score = np.mean([metric['macro avg']['f1-score'] for metric in full_metrics])

    print("- Accuracy: {:.3f}".format(accuracy))
    print("- Precision: {:.3f}".format(precision))
    print("- Recall: {:.3f}".format(recall))
    print("- F1-Score: {:.3f}".format(f1_score))


def build_log_reg_model(x_train, y_train):
    # Hyperparameter sweep
    param_grid = [
        {'solver': ['liblinear', 'saga'], 'penalty': ['l1', 'l2'], 'C': np.logspace(-4, 4, 10),
         'max_iter': [100, 200, 300]},
        {'solver': ['newton-cg', 'lbfgs', 'sag'], 'penalty': ['l2'], 'C': np.logspace(-4, 4, 10),
         'max_iter': [100, 200, 300]},
        {'solver': ['saga'], 'penalty': ['elasticnet'], 'C': np.logspace(-4, 4, 10), 'max_iter': [100, 200, 300],
         'l1_ratio': np.linspace(0, 1, 10)},
        {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'penalty': ['none'], 'max_iter': [100, 200, 300]}
    ]

    logreg = LogisticRegression()
    grid_search = GridSearchCV(logreg, param_grid, cv=5, n_jobs=-1)

    # Suppress warnings for convergence, fit failed or invalid solver for current algorithm
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FitFailedWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_

    return LogisticRegression(**best_params)


def build_lin_reg_model(x_train, y_train):
    pass


def build_knn_model(x_train, y_train):
    pass


def build_nn_model(x_train, y_train):
    pass
