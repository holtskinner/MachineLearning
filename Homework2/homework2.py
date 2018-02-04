import numpy as np
from scipy.io.matlab import loadmat
from scipy.spatial.distance import mahalanobis as mh


# x1 and x2 are points in 3-dimensions (1D Array with 3 items), I think?
def mahalanobis(x1, x2, covariance, mean, dimension):

    # Formula distance = √(x1 - x2)T ∑-1 (x1-x2)
    # numpy arrays have built in vector addition & subtraction!
    difference_vector = x1 - x2

    # Vector-Matrix Multiplication
    # first pair (numpy only allows two at a time)
    distance = np.dot(difference_vector, np.linalg.inv(covariance))

    distance = np.sqrt(np.dot(distance, difference_vector))

    return distance


def main():

    data = np.array(loadmat("./data_class3.mat")["Data"][0])

    x1 = np.array([-5.01, -8.12, -3.68])
    x2 = np.array([-5.43, -3.48, -3.54])

    cov_matrix = np.cov(data[0])
    distance = mahalanobis(x1, x2, cov_matrix, 0, 0)

    print(distance)
    print(mh(x1, x2, cov_matrix))
    # Each class of the data
    # for c in data:
    # print(c)

    # mean_vector = np.mean(c, axis=1)
    # print(f"{np.cov(c)}\n\n")

    return 0


main()