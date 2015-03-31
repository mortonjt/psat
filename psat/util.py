import numpy as np
import numbers


def check_array(mat):
    """
    Check if array is an numpy array or a scipy sparse array
    """
    pass


def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance
    If seed is None (or np.random), return the RandomState singleton used
    by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.

    Originally from scipy function
    https://github.com/scipy/scipy/blob/master/scipy/_lib/_util.py
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
