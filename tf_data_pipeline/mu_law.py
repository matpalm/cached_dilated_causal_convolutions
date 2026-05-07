import numpy as np

MU = 20


def mu_law_compress(x):
    """
    linear [-1, 1] to compressed [-1, 1]. ( no int quantisation )
    """
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * (np.log1p(MU * np.abs(x)) / np.log1p(MU))
    return y


def mu_law_expand(y):
    """
    compressed [-1, 1] back to a linear [-1, 1] range. ( again, no quantisation )
    """
    y = np.clip(y, -1.0, 1.0)
    x = np.sign(y) * (1 / MU) * (np.power(1 + MU, np.abs(y)) - 1)
    return x
