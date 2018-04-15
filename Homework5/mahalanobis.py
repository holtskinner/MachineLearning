import numpy as np


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
