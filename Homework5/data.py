import numpy as np
from scipy.io.matlab import loadmat


def load_data():

    train, test = loadmat("./test_train_data_class3.mat")["Data"][0][0]

    train = np.array(train[0])
    test = np.array(test[0])

    for i in range(train.shape[0]):
        train[i] = np.transpose(train[i])

    for i in range(test.shape[0]):
        test[i] = np.transpose(test[i])
    return train, test


def flatten_data(data, c):
    actual = np.array([], dtype=int)
    flat = np.zeros(2)

    # Flatten Test Array
    for i in range(c):
        for j in range(data[i].shape[0]):
            actual = np.append(actual, i)
            flat = np.vstack([flat, data[i][j]])

    flat = flat[1:len(flat) + 1]

    return flat, actual
