import numpy as np


def confusion_matrix(actual, predicted, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for a, p in zip(actual, predicted):
        cm[a, p] += 1

    acc = (actual == predicted).sum() / len(actual)

    return cm, acc
