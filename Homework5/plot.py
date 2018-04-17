import matplotlib.pyplot as plt
import numpy as np


def plot_data(r, theta, labels, c):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    for j in range(c):
        ax.scatter(theta[labels == j], r[labels == j],
                   label=f"Class {j}", alpha=0.5, edgecolors="none")

    ax.legend(range(c + 1))
    plt.show()
