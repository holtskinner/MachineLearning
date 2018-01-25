import scipy.io as sp
import numpy as np


def mean(matrix):

    total = 0
    num_values = 0

    for row in matrix:
        for item in row:
            num_values += 1
            total += item

    return 0 if num_values == 0 else total / (num_values)


def main():
    # DATA is a list of Matrices (3D Matrix!)
    data = sp.loadmat("./data_class4.mat")["Data"][0]

    print(np.mean(data[0]))

    m = mean(data[0])
    print(m)


main()
