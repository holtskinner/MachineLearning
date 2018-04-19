import numpy as np


def mah(x1, x2, cov):

    # Formula distance = √(x1 - x2)T ∑-1 (x1-x2)
    # numpy arrays have built in vector addition & subtraction!
    diff = x1 - x2

    # Vector-Matrix Multiplication
    # first pair (numpy only allows two at a time)

    if np.isscalar(cov):
        inv = 1 / cov
    else:
        inv = np.linalg.inv(cov)

    dist = np.dot(diff, inv)

    dist = np.dot(dist, diff)

    return dist


def discriminant(x, mean, covariance, dimension, prior):

    # g(x) = (-1/2) square(mahalanobis(x, mu)) - (d / 2)ln(2pi)
    #  - (1 / 2)ln(det(cov)) + ln(prior)

    a = (1 / 2) * mah(x, mean, covariance)

    # np.log is natural log
    b = (dimension / 2) * np.log(2 * np.pi)

    if np.isscalar(covariance):
        det = covariance
    else:
        det = np.linalg.det(covariance)
    c = (1 / 2) * np.log(det)

    d = np.log(prior)

    return -a - b - c + d
