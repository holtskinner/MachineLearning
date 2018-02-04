import numpy as np
from scipy.io.matlab import loadmat
from scipy.spatial.distance import mahalanobis as mh


# x1 and x2 are points in 3-dimensions (1D Array with 3 items), I think?
def mahalanobis(x1, x2, covariance):

    # Formula distance = √(x1 - x2)T ∑-1 (x1-x2)
    # numpy arrays have built in vector addition & subtraction!
    difference_vector = x1 - x2

    # Vector-Matrix Multiplication
    # first pair (numpy only allows two at a time)
    distance = np.dot(difference_vector, np.linalg.inv(covariance))

    # Not taking the square root because we need the "squared distance" for the discriminant"
    distance = np.dot(distance, difference_vector)

    return distance


def discriminant(x, mean, covariance, dimension, prior):

    # g(x) = (-1/2) square(mahalanobis(x, mu)) - (d / 2)ln(2pi)
    #  - (1 / 2)ln(det(cov)) + ln(prior)

    a = (1 / 2) * mahalanobis(x, mean, covariance)

    # np.log is natural log
    b = (dimension / 2) * np.log(2 * np.pi)

    c = (1 / 2) * np.log(np.linalg.det(covariance))

    d = np.log(prior)

    return -a - b - c + d


def main():

    data = np.array(loadmat("./data_class3.mat")["Data"][0])

    priors = np.array([0.6, 0.2, 0.2])
    test_points = np.array([[1, 3, 2], [4, 6, 1], [7, -1, 0], [-2, 6, 5]])

    dimensions = data.size
    mean_vectors = np.array([])
    cov_matrices = np.array([])

    # Each class of the data
    for c in data:
        np.append(mean_vectors, np.mean(c, axis=1))
        np.append(cov_matrices, np.cov(c))

    return 0


main()