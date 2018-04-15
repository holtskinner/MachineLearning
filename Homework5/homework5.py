import numpy as np
from scipy.io.matlab import loadmat
from max_likelihood import mle


def main():
    train_data, test_data = loadmat(
        "./test_train_data_class3.mat")["Data"][0][0]

    train_data = np.array(train_data[0])
    test_data = np.array(test_data[0])

    # For each class
    for c in train_data:
        mu, sigma = mle(c, )


if __name__ == "__main__":
    main()
