import numpy as np


def mu_estimate(data, mu_0, sigma_0, variance):
    mu_hat = np.mean(data)
    n = data.shape[0]

    ns = n * sigma_0

    a = ns / (ns + variance)
    b = variance / (ns + variance)

    mu_n = (a * mu_hat) + (b * mu_0)

    num = sigma_0 * variance
    den = n * sigma_0 + variance

    sigma_n = num / den

    return mu_n, sigma_n
