import numpy as np
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt
from scipy.io.matlab import loadmat
from scipy.spatial.distance import mahalanobis as mh
from mpl_toolkits.mplot3d import Axes3D


# x1 and x2 are points in 3-dimensions (1D Array with 3 items), I think?
def mahalanobis(x1, x2, covariance):

    # Formula distance = √(x1 - x2)T ∑-1 (x1-x2)
    # numpy arrays have built in vector addition & subtraction!
    difference_vector = x1 - x2

    # Vector-Matrix Multiplication
    # first pair (numpy only allows two at a time)
    distance = np.dot(difference_vector, np.linalg.inv(covariance))

    # Not taking the square root because we need the "squared distance" for the discriminant"
    distance = np.dot(distance, difference_vector)

    return distance


def discriminant(x, mean, covariance, dimension, prior):

    # g(x) = (-1/2) square(mahalanobis(x, mu)) - (d / 2)ln(2pi)
    #  - (1 / 2)ln(det(cov)) + ln(prior)

    a = (1 / 2) * mahalanobis(x, mean, covariance)

    # np.log is natural log
    b = (dimension / 2) * np.log(2 * np.pi)

    c = (1 / 2) * np.log(np.linalg.det(covariance))

    d = np.log(prior)

    return -a - b - c + d


def whiten(y, cov_matrix):

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Big Lambda inverse square root
    D = np.diag(1 / np.sqrt(eigenvalues))

    # Whitening Matrix
    W = np.dot(np.dot(eigenvectors, D), eigenvectors.T)

    return W


def part1():

    data = np.array(loadmat("./data_class3.mat")["Data"][0])

    priors = np.array([0.6, 0.2, 0.2])
    test_points = np.array([[1, 3, 2], [4, 6, 1], [7, -1, 0], [-2, 6, 5]])

    dimensions = data.size
    mean_vectors = []
    cov_matrices = []

    # Each class of the data
    for c in data:
        mean_vectors.append(np.mean(c, axis=1))
        cov_matrices.append(np.cov(c))

    for point in test_points:

        print(f"\nPoint: {point}")

        for i in range(dimensions):

            m = discriminant(point, mean_vectors[i], cov_matrices[i],
                             dimensions, priors[i])
            print(f"Class {i}: {m}")

    # TODO Figure out correct way to classify points

    return


def part2():

    # Part a, b, c
    # mean_vectors = np.array([[8, 2], [2, 8]])
    # cov_matrix = np.array([[4.1, 0], [0, 2.8]])

    # y = np.random.standard_normal(1000)

    u1 = [8, 2]
    u2 = [2, 8]
    cov1 = [[4.1, 0], [0, 2.8]]
    cov2 = cov1
    prior1 = .8

    data_points = np.random.multivariate_normal(u1, cov1, 50).T
    # print(xy)

    X, Y = np.meshgrid(data_points[0], data_points[1])
    xy = np.column_stack([X.flat, Y.flat])

    z = mn.pdf(xy, mean=u1, cov=cov1)
    Z = z.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    plt.show()

    # print(x)

    # Part d
    # cov_matrix[0, 1] = 0.4
    # cov_matrix[1, 0] = 0.4

    # # Part e
    # cov_matrix1 = np.array([[2.1, 1.5], [1.5, 3.8]])
    # cov_matrix2 = cov_matrix

    return


part1()
part2()
