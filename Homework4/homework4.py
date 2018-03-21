import numpy as np
from data import data


def max_likelihood(x, n):

    x = x.T
    mu = np.sum(x, axis=0) / n

    d = x - mu

    dt = np.multiply(d, d)

    sigma = np.sum(dt, axis=0) / (n - 1)

    return mu, sigma


num_features = 3
num_classes = 3
num_samples = 10

data_by_features = np.transpose(data, (2, 1, 0))

data_by_class = np.transpose(data, (1, 2, 0))

for i in range(num_classes):
    mu, sigma = max_likelihood(data_by_class[0, i], num_samples)
    print(f"Class {i}")
    print("μ       σ")
    print(mu, sigma)
    print()
