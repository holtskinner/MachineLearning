import numpy as np


def cart2pol(data):
    x = data.T[0]
    y = data.T[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    data.T[0] = rho
    data.T[1] = phi
    return rho, phi, data


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
