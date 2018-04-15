import numpy as np
from scipy.io.matlab import loadmat
from max_likelihood import mle
from mahalanobis import discriminant


def load_data(train, test):

    # train, test =
    # print(train, test)

    train = np.array(train[0])
    test = np.array(test[0])

    for i in range(train.shape[0]):
        train[i] = np.transpose(train[i])

    for i in range(test.shape[0]):
        test[i] = np.transpose(test[i])

    return train, test


def main():

    train, test = loadmat("./test_train_class3.mat")["Data"][0][0]

    means = np.empty(train.shape[0])
    covs = np.empty(train.shape[0])

    # Part A
    for i in range(train.shape[0]):
        means[i], covs[i] = mle(train[i])

    # Part B
    prior = 1 / 3
    d = train.shape[1]
    print(d)

    for i in range(train.shape[0]):
        m = discriminant(train[i], means[i], covs[i], d, prior)


if __name__ == "__main__":
    main()
