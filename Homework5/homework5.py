import numpy as np
from scipy.io.matlab import loadmat
from max_likelihood import mle


def load_data():

    train_data, test_data = loadmat(
        "./test_train_data_class3.mat")["Data"][0][0]

    train_data = np.array(train_data[0])
    test_data = np.array(test_data[0])

    for i in range(train_data.shape[0]):
        train_data[i] = np.transpose(train_data[i])

    for i in range(test_data.shape[0]):
        test_data[i] = np.transpose(test_data[i])

    return train_data, test_data


def main():

    train_data, test_data = load_data()

    # For each class
    for c in train_data:

        mean, covariance = mle(c)


if __name__ == "__main__":
    main()
