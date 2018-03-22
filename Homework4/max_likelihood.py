import numpy as np
from data import data


def print_mu_sigma(mu, sigma):
    print("μ")
    print(mu)
    print("σ")
    print(sigma)
    print()


def max_likelihood(x, n):

    x = x.T
    mu = 0

    for k in range(n):
        mu += x[k]

    mu /= n

    sigma = 0

    for k in range(n):
        d = x[k] - mu
        sigma += d * d.T

    sigma /= (n)

    return mu, sigma


def main():
    num_features = 3
    num_classes = 3
    num_samples = 10

    data_by_features = np.transpose(data, (2, 1, 0))

    data_by_class = np.transpose(data, (1, 2, 0))

    class1 = data_by_class[0]
    print("Part A\n")

    for i in range(num_classes):
        mu, sigma = max_likelihood(data_by_class[0, i], num_samples)
        print(f"Feature {i}")
        print_mu_sigma(mu, sigma)

    print("Part B\n")
    combinations = np.array([
        [class1[0], class1[1]],
        [class1[0], class1[2]],
        [class1[1], class1[2]]
    ])

    for b in combinations:
        mu, sigma = max_likelihood(b, num_samples)
        print_mu_sigma(mu, sigma)

    print("Part C\n")
    mu, sigma = max_likelihood(class1, num_samples)
    print_mu_sigma(mu, sigma)

    print("Part D\n")
    mu, sigma = max_likelihood(data_by_class[1], num_samples)
    print_mu_sigma(mu, sigma)
