import numpy as np


def mle(x):

    n = x.shape[0]
    mu = np.sum(x, axis=0) / n
    sigma = np.subtract(x, mu)
    sigma = np.dot(sigma, np.transpose(sigma)) / n

    return mu, sigma
