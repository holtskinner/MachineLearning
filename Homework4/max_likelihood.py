import numpy as np


def max_likelihood(x, n):

    x = x.T
    mu = 0

    for k in range(n):
        mu += x[k]

    mu /= n

    sigma = 0

    for k in range(n):
        d = x[k] - mu
        sigma += d * np.transpose(d)

    sigma /= (n)

    return mu, sigma
