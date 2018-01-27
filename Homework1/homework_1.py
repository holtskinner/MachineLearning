import scipy.io as sp
import numpy as np


def mean(matrix):

    # µ E(x)
    mean_vector = []
    num_features = 0
    num_observations = 0

    for row in matrix:

        total = 0
        num_features = 0

        for feature in row:
            num_features += 1
            total += feature

        mean_vector.append(total / num_features)
        num_observations += 1

    return (mean_vector, num_features, num_observations)


def covariance(m):

    mean_vector, num_features, num_observations = mean(m)

    # Create empty matrix size d * d
    cov_matrix = np.zeros((num_observations, num_observations))

    # i & j coordinates of cov_matrix
    # k - features in data
    for i in range(num_observations):

        for j in range(num_observations):

            for k in range(num_features):
                cov_matrix[i][j] += (m[i][k] - mean_vector[i]) * (
                    m[j][k] - mean_vector[j])

            cov_matrix[i][j] /= num_features

    return (cov_matrix, mean_vector)


def main():

    # DATA is a list of Matrices (3D Matrix!)
    data = sp.loadmat("./data_class4.mat")["Data"][0]

    print("Mean & Covariance")

    # matrix Because class is a reserved word
    for index, matrix in enumerate(data):
        calculated_cov, mean_vector = covariance(matrix)
        numpy_cov = np.cov(matrix)
        numpy_mean_x = np.mean(matrix[0])
        numpy_mean_y = np.mean(matrix[1])
        print("\n---------------")
        print(f"Class {index}:")
        print("---------------\n")
        print("Covaraiance")
        print("By Hand:")
        print(calculated_cov)
        print("Built-In:")
        print(numpy_cov)
        print("\nMean")
        print("X")
        print(f"By Hand: {mean_vector[0]}")
        print(f"Built-In {numpy_mean_x}")
        print("Y")
        print(f"By Hand: {mean_vector[1]}")
        print(f"Built-In {numpy_mean_y}")

    # for matrix in data:
    #     eigen = np.linalg.eigvals(matrix)
    #     print(eigen)


main()
