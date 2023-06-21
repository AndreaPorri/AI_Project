import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset_function import load_data_from_file, create_splits_unbalanced, normalize_data



def predict_and_r2_score(net, X, y, minibatch_size=None):
    """Make prediction and compute the R2 score (it supports mini-batches).

    Args:
        net: the neural network.
        X: dataset on which predictions are performed.
        y: ground truth targets.
        minibatch_size (optional): size of the mini-batches.

    Returns:
        R2 score and the network output.
    """

    device = next(net.parameters()).device  # we assume that all the network parameters are on the same device
    n = X.shape[0]
    r2_status = None
    t = 0
    outputs = []

    if minibatch_size is None:
        minibatch_size = n

    with torch.no_grad():
        training_mode = net.training
        net.eval()

        while True:  # loop on mini-batches
            f = t
            t = min(f + minibatch_size, n)
            X_minibatch = X[f:t, :].to(device)
            y_minibatch = y[f:t].to(device)

            output = net(X_minibatch)
            outputs.append(output)

            _, r2, r2_status = r2_score(output, y_minibatch, update_from=r2_status)

            if t == n:
                break

        if training_mode:
            net.train()

        return r2, torch.cat(outputs, dim=0)


def r2_score(outputs, targets, update_from=None):
    """Compute the R2 score of the provided network outputs with respect to given targets.

    Args:
        outputs: the outputs of the network on a batch of data.
        targets: the ground truth targets of the data.
        update_from (optional): a 4-element list/tuple, composed of those quantities that are needed to model
            an initial status of the R2 computation; the computation of the R2 will start from this status.

    Returns:
        R2 score computed on the given data, computed also considering the initial status, and the status (4-element
            tuple) needed for future updates of this score.
    """

    sum_errors = torch.sum(torch.pow(outputs - targets, 2)).item()
    sum_squared_targets = torch.sum(torch.pow(targets, 2)).item()
    sum_targets = torch.sum(targets).item()
    n = outputs.shape[0]
    r2 = 1. - sum_errors / (sum_squared_targets - (sum_targets ** 2) / n)

    if update_from is not None:
        sum_errors += update_from[0]
        sum_squared_targets += update_from[1]
        sum_targets += update_from[2]
        n += update_from[3]
        r2_updated = 1. - sum_errors / (sum_squared_targets - (sum_targets ** 2) / n)
    else:
        r2_updated = r2

    status = (sum_errors, sum_squared_targets, sum_targets, n)

    return r2, r2_updated, status


def train(net, X_train, y_train, X_val, y_val, epochs=2000, lr=0.001, minibatch_size=32):
    """Train a neural network for multiple epochs.

    Args:
        net: the neural network.
        X_train: training data (one-example-per-row).
        y_train: training targets.
        X_val: validation data (one-example-per-row).
        y_val: validation targets.
        epochs (optional): number of epochs.
        lr (optional): learning rate.
        minibatch_size (optional): size of the training mini-batches.

    Returns:
        The two arrays with the R2s on training and validation data computed during the training epochs.
    """

    def rmse_loss(o, y):
        return torch.sqrt(torch.nn.functional.mse_loss(o, y, reduction='mean'))

    net.train()  # telling the network we are going to train it (and not to simply evaluate it)
    loss = rmse_loss  # defining the loss function
    optimizer = torch.optim.Adam(net.parameters(), lr)  # defining the way we are going to update the net parameters
    device = next(net.parameters()).device  # we assume that all the network parameters are on the same device

    n = X_train.shape[0]
    best_r2_val = None
    r2s_train = np.zeros(epochs)
    r2s_val = np.zeros(epochs)

    for e in range(0, epochs):  # loop on epochs
        loss_value = 0.
        r2_status = None
        t = 0
        nb = 0

        while True:  # loop on mini-batches
            optimizer.zero_grad()  # clearing the previously computed gradients

            f = t
            t = min(f + minibatch_size, n)
            X_minibatch = X_train[f:t, :].to(device)
            y_minibatch = y_train[f:t].to(device)

            outputs = net(X_minibatch)  # going forward, "net" is a callable object
            loss_value_on_minibatch = loss(outputs, y_minibatch)  # RMSE

            with torch.no_grad():
                r2_train_on_minibatch, r2_train, r2_status = r2_score(outputs, y_minibatch, update_from=r2_status)

                print("\tminibatch: {}, loss_train: {:.4f}, "
                      "r2_train: {:.2f}".format(nb + 1, loss_value_on_minibatch, r2_train_on_minibatch))

                loss_value += (loss_value_on_minibatch.item() ** 2) * (t - f)  # needed to estimate the train loss

            loss_value_on_minibatch.backward()  # going backward
            optimizer.step()  # updating model parameters

            nb += 1
            if t == n:
                break

        torch.save(net.state_dict(), 'net.pth')

        r2_val, _ = predict_and_r2_score(net, X_val, y_val, minibatch_size=minibatch_size)
        found_best = False
        if best_r2_val is None or r2_val > best_r2_val:
            best_r2_val = r2_val
            found_best = True
            torch.save(net.state_dict(), 'net_best.pth')

        loss_value = np.sqrt(loss_value / n)
        print("epoch: {}, loss_train: {:.4f}, "
              "r2_train: {:.2f}, r2_val: {:.2f}".format(e + 1, loss_value, r2_train, r2_val)
              + (" (best)" if found_best else ""))

        r2s_train[e] = r2_train
        r2s_val[e] = r2_val

    return r2s_train, r2s_val


def plot(r2s_train, r2s_val, output_test, test_y):
    """Plot the R2 score computed on the training and validation data during the training stage.

    Args:
        r2s_train: the array with the training R2 scores (during the training epochs).
        r2s_val: the arrays with the validation R2 scores (during the training epochs).
        output_test: the prediction on the test data.
        test_y: the target of the test data
    """

    plt.figure()
    plt.plot(r2s_train, label='Training Data')
    plt.plot(r2s_val, label='Validation Data')
    plt.ylabel('R2 Score')
    plt.xlabel('Epochs')
    plt.ylim((-1.1, 1.1))
    plt.legend(loc='lower right')
    plt.savefig('training_stage.pdf')

    plt.figure()
    plt.plot(output_test, test_y, '*')
    plt.ylabel('Target Price')
    plt.xlabel('Predicted Price')
    plt.savefig('test_stage.pdf')

# entry point
if __name__ == "__main__":
    my_device = torch.device("cpu")  # device on which computations will be performed ("cpu", "cuda:0", "cuda:1", ...)

    # dataset can be downloaded from:
    # https://archive.ics.uci.edu/ml/machine-learning-databases/housing
    # files to download: housing.data (dataset), housing.names (description of the dataset, have a look at it!)
    data_X, data_y = load_data_from_file('housing.data')

    # splitting
    data_X_train, data_y_train, data_X_val, data_y_val, data_X_test, data_y_test = \
        create_splits_unbalanced(data_X, data_y, train_val_fractions=[0.7, 0.15])

    # normalizing data
    data_X_train, m, s = normalize_data(data_X_train)
    data_X_val, _, _ = normalize_data(data_X_val, m, s)
    data_X_test, _, _ = normalize_data(data_X_test, m, s)

    # ensure that we keep track of the mean and std used to normalize the data
    torch.save([m, s], 'normalizers.pth')

    # network (you can customize it)
    my_net = torch.nn.Sequential(
        torch.nn.Linear(13, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 1)
    ).to(my_device)

    # training the network
    train_r2s, val_r2s = train(my_net,
                               data_X_train, data_y_train,
                               data_X_val, data_y_val,
                               epochs=1, lr=0.001, minibatch_size=64)

    # loading the best network from disk
    my_net.load_state_dict(torch.load('net_best.pth', map_location=my_device))

    # making predictions with the best found network
    train_r2, _ = predict_and_r2_score(my_net, data_X_train, data_y_train, minibatch_size=16)
    val_r2, _ = predict_and_r2_score(my_net, data_X_val, data_y_val, minibatch_size=16)
    test_r2, test_output = predict_and_r2_score(my_net, data_X_test, data_y_test, minibatch_size=16)
    print("train r2: {}%".format(train_r2))
    print("val r2: {}%".format(val_r2))
    print("test r2: {}%".format(test_r2))

    # creating final plot
    plot(train_r2s, val_r2s, test_output, data_y_test)

