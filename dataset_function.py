import torch
import numpy as np
import matplotlib.pyplot as plt

def load_data_from_file(file):
    """Load data from the Boston Housing data file, creating the X and y tensors.

    Args:
        file: the name of the CSV file with the data.

    Returns:
        The NumPy tensors X,y (data and targets).
    """

    # loading data from file (regression targets are on the last column)
    Xy = np.loadtxt(file, dtype=np.float32)

    # splitting data and targets
    X = Xy[:, 0:-1]
    y = Xy[:, -1].reshape((X.shape[0], 1))
    return X, y

def create_splits_unbalanced(X, y, train_val_fractions=(0.5, 0.25), randomize=True):
    """Creates (randomized) training, validation, test data splits.

    Args:
        X: dataset (one example per row).
        y: ground truth labels (vector).
        train_val_fractions: the fraction of data to be used in the training and validation sets, respectively.
        randomize (optional): randomize the data before splitting them.

    Returns:
        The NumPy tensors X,y (data and targets) for training, validation and test sets, respectively.
    """

    # randomizing data
    n = X.shape[0]
    if randomize:
        indices = np.arange(n)
        np.random.shuffle(indices)
        X = X[indices, :]
        y = y[indices]

    # collecting indices of the examples that will fall into the training, validation and test sets
    n = X.shape[0]
    a = round(n * train_val_fractions[0])
    b = a + round(n * train_val_fractions[1])

    train_set_indices = np.arange(0, a)
    val_set_indices = np.arange(a, b)
    test_set_indices = np.arange(b, n)

    # splitting into training, validation and test sets
    X_train = torch.from_numpy(X[train_set_indices, :])
    y_train = torch.from_numpy(y[train_set_indices])
    X_val = torch.from_numpy(X[val_set_indices, :])
    y_val = torch.from_numpy(y[val_set_indices])
    X_test = torch.from_numpy(X[test_set_indices, :])
    y_test = torch.from_numpy(y[test_set_indices])

    return X_train, y_train, X_val, y_val, X_test, y_test

def normalize_data(X, mean=None, std=None):
    """Normalize the data to a given mean and standard deviation (or to null-mean and unit-std).

    Args:
        X: the data to be normalized (one-example-per-row).
        mean (optional): the array with the means to be used in the normalization.
        std (optional): the array with the standard deviations to be used in the normalization.

    Returns:
        The normalized dataset and the arrays with the means and standard deviations that were used.
    """

    if mean is None:
        mean = torch.mean(X, dim=0)
    if std is None:
        std = torch.std(X, dim=0)
    return (X - mean) / std, mean, std
