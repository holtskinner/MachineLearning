from numpy import exp, abs, angle


def pol2cart(r, theta):
    return r * exp(1j * theta)


def cart2pol(z):
    return (abs(z), angle(z))
