from __future__ import division
from future.builtins import range
import numpy as np
import copy
from psat.util import check_random_state


def permutation_ttest(mat, cats, axis=0, permutations=10000,
          chunk_size=10, equal_var=True, random_state=None):
    """
    Calculates the T-test for the means of TWO INDEPENDENT samples of
    scores using permutation methods

    This test is an equivalent to scipy.stats.ttest_ind, except it
    doesn't require the normality assumption since it uses a
    permutation test.  This function is only called from ttest_ind if
    the p-value is calculated using a permutation test

    Parameters
    ----------
    mat : array_like
        This contains all of the data values for both groups.
    cats: array_like
        This array must be 1 dimensional and have the same size as
        the length of the specified axis.  This encodes information
        about which category each element of the mat array belongs to
    axis : int, optional
        Axis can equal None (ravel array first), or an integer
        (the axis over which to operate on a and b).
    permutations: int
        Number of permutations used to calculate p-value
    equal_var: bool
        If false, a Welch's t-test is conducted.  Otherwise,
        a ordinary t-test is conducted
    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    Returns
    -------
    t-stat : float or array
        The calculated t-statistic.
    """
    if axis == 0:
        mat = mat.transpose()
    if mat.ndim < 2:  # Handle 1-D arrays
        mat = mat.reshape((1, len(mat)))
    perms = _init_summation_index(cats, permutations=permutations,
                                  random_state=random_state)
    num_cats = 2
    _, c = perms.shape

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
        svar = ((tot[idx+1] - 1) * _samp_vars[:, idx+1] +
                (tot[idx] - 1) * _samp_vars[:, idx]) / df
        denom = np.sqrt(svar * (1.0 / tot[idx+1] + 1.0 / tot[idx]))

    t_stat = np.divide(_avgs[:, idx] - _avgs[:, idx+1], denom)

    # Calculate the p-values
    cmps = abs(t_stat[:, 1:].transpose()) >= abs(t_stat[:, 0])
    pvalues = (cmps.sum(axis=0) + 1.) / (permutations + 1.)

    return t_stat

def permutation_f_oneway(mat, cats, **kwds):
    """
    Performs a 1-way ANOVA.

    The one-way ANOVA tests the null hypothesis that two or more groups have
    the same population mean.  The test is applied to samples from two or
    more groups, possibly with differing sizes.

    Parameters
    ----------
    mat : np.ndarray
        Contingency table
    cats : array_like
        Categorical vector
    axis : int, optional
        Axis to conduct permutation test on.
    permutations : int, optional
        If permutations > 0, then a permutation test will be conducted to
        calculate the p-values
    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    Returns
    -------
    F-value : float
        The computed F-values of the test.
    """

    params = {'permutations':10000, 'random_state':0, 'axis':0}
    for key in ('permutations', 'random_state', 'axis'):
        params[key] = kwds.get(key, params[key])
    permutations = params['permutations']
    random_state = params['random_state']
    axis = params['axis']

    random_state = check_random_state(random_state)
    if axis == 0:
        mat = mat.transpose()
    if len(mat.shape) < 2:
        mat = mat.reshape((1, len(mat)))
    r, c = mat.shape
    num_cats = len(np.unique(cats))
    f_stat = np.zeros((r, num_cats*(permutations+1)))
    copy_cats = copy.deepcopy(cats)

    perms = _init_summation_index(copy_cats, permutations)
    n_samp, c = perms.shape
    mat2 = np.multiply(mat, mat)
    S = mat.sum(axis=1)
    SS = mat2.sum(axis=1)
    sstot = SS - np.multiply(S,S) / n_samp
    sstot = sstot.reshape((len(sstot),1))
    # Create index to sum the ssE together
    sum_groups = np.arange(num_cats, dtype=np.int32) // num_cats
    _sum_idx = _init_summation_index(sum_groups)

    # Perform matrix multiplication on data matrix
    # and calculate sums and squared sums and sum of squares
    _sums = np.dot(mat, perms)
    _sums2 = np.dot(np.multiply(mat, mat), perms)

    tot = perms.sum(axis=0)
    ss = _sums2 - np.multiply(_sums, _sums)/tot
    sserr = np.dot(ss, _sum_idx)
    sstrt = sstot - sserr
    dftrt = num_cats - 1
    dferr = np.dot(tot, _sum_idx) - num_cats
    f_stat = np.ravel((sstrt / dftrt) / (sserr / dferr))

    cmps = f_stat[:, 1:].transpose() >= f_stat[:, 0]
    pvalues = (cmps.sum(axis=0) + 1.) / (permutations + 1.)

    return f_stat

#################################################################
#                       Helper functions                        #
#################################################################
def _init_summation_index(cats, permutations, random_state=None):
    """
    Creates a matrix filled with category permutations

    cats: numpy.array
       List of class assignments
    """
    random_state = check_random_state(random_state)
    c = len(cats)
    num_cats = len(np.unique(cats))  # Number of distinct categories
    copy_cats = copy.deepcopy(cats)
    perms = np.array(np.zeros((c, num_cats*(permutations+1)),
                              dtype=cats.dtype))
    for m in xrange(permutations+1):
        for i in xrange(num_cats):
            perms[:, num_cats*m+i] = (copy_cats == i).astype(cats.dtype)
        random_state.shuffle(copy_cats)
    return perms
