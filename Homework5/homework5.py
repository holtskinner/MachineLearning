import numpy as np
from scipy.io.matlab import loadmat
from mahalanobis import discriminant
from sklearn.metrics import confusion_matrix


def load_data():

    train, test = loadmat("./test_train_data_class3.mat")["Data"][0][0]

    train = np.array(train[0])
    test = np.array(test[0])

    for i in range(train.shape[0]):
        train[i] = np.transpose(train[i])

    for i in range(test.shape[0]):
        test[i] = np.transpose(test[i])
    return train, test


def main():

    train, test = load_data()

    prior = 1 / 3

    # Num Classes
    c = train.shape[0]
    # Num Dimensions
    d = train[0].shape[1]

    means = np.empty((c, d))
    covs = np.empty(((c, d, d)))

    for i in range(c):
        means[i] = np.mean(train[i], axis=0)
        covs[i] = np.cov(train[i], rowvar=0)

    expected = np.array([], dtype=int)
    actual = np.array([], dtype=int)

    flat_test = np.zeros(2)

    # Flatten Test Array
    for i in range(c):
        for j in range(test[i].shape[0]):
            expected = np.append(expected, i)
            flat_test = np.vstack([flat_test, test[i][j]])

    flat_test = flat_test[1:101]
    disc_values = np.zeros((100, 3))

    for i, point in enumerate(flat_test):
        for j in range(c):
            m = discriminant(point, means[j], covs[j], d, prior)
            disc_values[i, j] = m

    actual = np.argmax(disc_values, axis=1)

    print(expected)
    print(actual)


if __name__ == "__main__":
    main()
