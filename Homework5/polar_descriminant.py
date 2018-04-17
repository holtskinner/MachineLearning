import numpy as np


def mu_estimate(data, mu_0, sigma_0, variance):
    mu_hat = np.mean(data)
    n = data.shape[0]

    ns = n * sigma_0

    a = ns / (ns + variance)
    b = variance / (ns + variance)

    mu_n = (a * mu_hat) + (b * mu_0)

    return mu_n
