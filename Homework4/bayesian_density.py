import numpy as np
from data import data
from max_likelihood import max_likelihood
import matplotlib.pyplot as plt


def triangle(x, mu, delta):
    d = np.abs(x - mu)

    if (d >= delta):
        return 0

    return(delta - d) / (delta ** 2)


def density(x):
    n = x.shape[0]
    d = np.empty(n)
    mu, sigma = max_likelihood(x, n)

    for i in range(n):
        d[i] = triangle(x[i], mu, np.sqrt(sigma))
    return np.sum(d), d, mu, sigma


def main():

    data_by_class = np.transpose(data, (1, 2, 0))

    x = data_by_class[1, 1]

    d = density(x)

    print("Density")
    print(d[0])
    print(d[1])
    plt.plot(x, d[1], 'bo')
    plt.xlabel("x2")
    plt.ylabel("Density")
    plt.axvline(x=d[2])
    plt.legend()
    plt.show()


main()
