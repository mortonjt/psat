import numpy as np
import scipy as sp



def ttest(mat, cats, axis=0, permutations=10000,
          chunk_size=10, equal_var=True, random_state=None):
    """
    Calculates the T-test for the means of TWO INDEPENDENT samples of scores
    using permutation methods

    This test is an equivalent to scipy.stats.ttest_ind, except it doesn't require the
    normality assumption since it uses a permutation test.  This function is only
    called from ttest_ind if the p-value is calculated using a permutation test

    Parameters
    ----------
    mat : array_like
        This contains all of the data values for both groups.
    cats: array_like
        This array must be 1 dimensional and have the same size as
        the length of the specified axis.  This encodes information
        about which category each element of the mat array belongs to
    axis : int, optional
        Axis can equal None (ravel array first), or an integer (the axis
        over which to operate on a and b).
    permutations: int
        Number of permutations used to calculate p-value
    equal_var: bool
        If false, a Welch's t-test is conducted.  Otherwise,
        a ordinary t-test is conducted
    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    Returns
    -------
    t : float or array
        The calculated t-statistic.
    prob : float or array
        The two-tailed p-value.
    """
    if axis == 0:
        mat = mat.transpose()
    if len(mat.shape) < 2:  # Handle 1-D arrays
        mat = mat.reshape((1, len(mat)))
    perms = _init_categorical_perms(cats, permutations=permutations, random_state=random_state)
    num_cats = 2
    _, c = perms.shape
    permutations = (c - num_cats) / num_cats

    # Perform matrix multiplication on data matrix
    # and calculate sums and squared sums
    _sums = np.dot(mat, perms)
    _sums2 = np.dot(np.multiply(mat, mat), perms)

    # Calculate means and sample variances
    tot = perms.sum(axis=0)
    _avgs = _sums / tot
    _avgs2 = _sums2 / tot
    _vars = _avgs2 - np.multiply(_avgs, _avgs)
    _samp_vars = np.multiply(tot, _vars) / (tot-1)
    idx = np.arange(0, (permutations+1) * num_cats, num_cats, dtype=np.int32)

    # Calculate the t statistic
    if not equal_var:
        denom = np.sqrt(np.divide(_samp_vars[:, idx+1], tot[idx+1]) +
                         np.divide(_samp_vars[:, idx], tot[idx]))
    else:
        df = tot[idx] + tot[idx+1] - 2
        svar = ((tot[idx+1] - 1) * _samp_vars[:, idx+1] + (tot[idx] - 1) * _samp_vars[:, idx]) / df
        denom = np.sqrt(svar * (1.0 / tot[idx+1] + 1.0 / tot[idx]))

    t_stat = np.divide(_avgs[:, idx] - _avgs[:, idx+1], denom)

    # Calculate the p-values
    cmps = abs(t_stat[:, 1:].transpose()) >= abs(t_stat[:, 0])
    pvalues = (cmps.sum(axis=0) + 1.) / (permutations + 1.)

    return t_stat[:, 0], pvalues


## Helper functions

def _init_summation_index(cats, permutations):
    """
    Creates a matrix filled with category permutations

    cats: numpy.array
       List of class assignments
    """
    c = len(cats)
    num_cats = len(np.unique(cats))  # Number of distinct categories
    copy_cats = copy.deepcopy(cats)
    perms = np.array(np.zeros((c, num_cats), dtype=cats.dtype))
    for m in range(permutations+1):
        for i in range(num_cats):
            perms[:,num_cats*m+i] = (copy_cats == i).astype(cats.dtype)
        np.random.shuffle(copy_cats)
    return perms
