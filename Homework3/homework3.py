import numpy as np


def bbn(prob_of: dict, given: dict):

    probability = 1

    key = list(prob_of.keys())[0]

    if key == "x":

        # prob_of["x"] == 0 if salmon, 1 if sea bass
        x = prob_of["x"]
        a = given["a"]
        probability *= x_a[x, a]

        b = given["b"]
        probability *= x_b[x, b]

        c = given["c"]
        probability *= c_x[x, c]

        d = given["d"]
        probability *= d_x[x, d]

        probability /= .5

    elif key == "a":

        a = prob_of["a"]

        b = given["b"]
        probability *= x_b[0, b] + x_b[1, b]

        c = given["c"]
        probability *= c_x[0, c] + c_x[1, c]

        d = given["d"]
        probability *= d_x[0, d] + d_x[1, d]

        probability /= (.5 * .5 * .33 * .5)

    elif key == "b":

        b = prob_of["b"]

        a = given["a"]
        probability *= x_a[0, a] + x_a[1, a]

        c = given["c"]
        probability *= c_x[0, c] + c_x[1, c]

        d = given["d"]
        probability *= d_x[0, d] + d_x[1, d]

        probability /= (.25 * .5 * .33 * .5)

    return probability


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

# Part a
prob_1 = {"x": 0}
given_1 = {"a": 0, "b": 1, "c": 0, "d": 1}
part_a = bbn(prob_1, given_1) * 100

print("Part A")
print(f"Probability of Salmon: {part_a:.3f}%\n")

# Part b

prob_2 = {"a": 0}
given_2 = {"b": 1, "c": 0, "d": 1}

part_b = bbn(prob_2, given_2)
print("Part B")
print(f"Probability of Winter: {part_b:.3f}%")

prob_2 = {"a": 1}

part_b = bbn(prob_2, given_2)
print(f"Probability of Spring: {part_b:.3f}%")

prob_2 = {"a": 2}

part_b = bbn(prob_2, given_2)
print(f"Probability of Summer: {part_b:.3f}%")

prob_2 = {"a": 3}

part_b = bbn(prob_2, given_2)
print(f"Probability of Autumn: {part_b:.3f}%\n")

prob_3 = {"b": 0}
given_3 = {"a": 2, "c": 2, "d": 0}

part_c = bbn(prob_3, given_3)
print("Part C")
print(f"Probability of North Atlantic: {part_c:.3f}%\n")
