import itertools

import numpy as np


def argmax_ndim(arg_array):
    return np.unravel_index(arg_array.argmax(), arg_array.shape)


def argmin_ndim(arg_array):
    return np.unravel_index(arg_array.argmin(), arg_array.shape)


def enumerated_product(*args):
    yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))
