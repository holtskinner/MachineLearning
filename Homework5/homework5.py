import numpy as np
from scipy.io.matlab import loadmat
from max_likelihood import mle


def load():

    train, test = loadmat(
        "./test_train_class3.mat")["Data"][0][0]

    train = np.array(train[0])
    test = np.array(test[0])

    for i in range(train.shape[0]):
        train[i] = np.transpose(train[i])

    for i in range(test.shape[0]):
        test[i] = np.transpose(test[i])

    return train, test


def main():

    train, test = load()

    # Part a
    for c in train:
        mean, cov = mle(c)


if __name__ == "__main__":
    main()
