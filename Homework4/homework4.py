import numpy as np


def max_likelihood(x, n):

    mu = np.sum(x) / n
    sigma = np.sum(np.dot(x, mu)) / (n - 1)

    return mu, sigma


num_features = 3
num_classes = 3
num_samples = 10

# Dimensions (Num Samples X num classes X num features)
data = np.array([
    # Sample 1
    [
        # [x1, x2, x3]
        # Class w1
        [0.42, -0.087, 0.58],

        # Class w2
        [-0.4, 0.58, 0.089],

        # Class w3
        [0.83, 1.6, -0.014]
    ],
    # Sample 2
    [
        [-0.2, -3.3, -3.4],
        [-0.31, 0.27, -0.04],
        [1.1, 1.6, 0.48]
    ],
    [
        [1.3, -0.32, 1.7],
        [0.38, 0.055, -0.035],
        [-0.44, -0.41, 0.32]
    ],
    [
        [0.39, 0.71, 0.23],
        [-0.15, 0.53, 0.011],
        [0.047, -0.45,  1.4]
    ],
    [
        [-1.6, -5.3, -0.15],
        [-0.35, 0.47, 0.034],
        [0.28, 0.35, 3.1]
    ],
    [
        [-0.029, 0.89, -4.7],
        [0.17, 0.69, 0.1],
        [-0.39, -0.48, 0.11]
    ],
    [
        [-0.23, 1.9, 2.2],
        [-0.011, 0.55, -0.18],
        [0.34, -0.079, 0.14]
    ],
    [
        [0.27, -0.3, -0.87],
        [-0.27, 0.61, 0.12],
        [-0.3, -0.22, 2.2]
    ],
    [
        [-1.9, 0.76, -2.1],
        [-0.065, 0.49, 0.0012],
        [1.1, 1.2, -0.46]
    ],
    [
        [0.87, -1.0, -2.6],
        [-0.12, 0.054, -0.063],
        [0.18, -0.11, -0.49]
    ]
])

data_by_features = data.T

data_by_class = np.zeros((num_classes, num_features, num_samples))

for i in range(num_classes):
    for j in range(num_features):
        data_by_class[i, j] = data_by_features[j, i]

print(data_by_class)
