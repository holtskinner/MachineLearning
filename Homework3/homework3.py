import numpy as np


def main():

    # Fish given Season
    # Winter, Spring, Summer, Autumn
    x_a = np.array([
        # Salmon
        [.9, .3, .4, .8],
        # Sea Bass
        [.1, .7, .6, .2]
    ])

    # Fish given Location
    # North, South
    x_b = np.array([
        # Salmon
        [.65, .25],
        # Sea Bass
        [.35, .75]
    ])

    # Lightness given Fish
    # Light, Medium, Dark
    c_x = np.array([
        # Salmon
        [.33, .33, .34],
        #Sea Bass
        [.8, .1, .1]
    ])

    # Thickness given Fish
    # Wide, Thin
    d_x = np.array([
        # Salmon
        [.4, .6],
        # Sea Bass
        [.95, .05]
    ])
    return


main()