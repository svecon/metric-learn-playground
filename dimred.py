#    https://github.com/isido/dimred/blob/master/dimred/trustcont.py
#    Dimensionality Reduction Tools
#    Copyright (C) 2010 Ilja Sidoroff

import math
import numpy

def d_euclidean(a,b):
    """Returns euclidean distance between vectors a and b"""

    return math.sqrt(sum([ (x - y)**2 for (x,y) in zip (a,b) ]))


def distance_matrix(x, distf=None):
    """Returns distance matrix of pairwise distances between vectors
    stored in array a
    distf is function used to calculate distance between vectors,
    if None, euclidean distance is used
    """

    if distf == None:
        distf = d_euclidean

    vectors = x.shape[0]
    d = numpy.zeros([vectors, vectors])

    for i in range(vectors):
        for j in range(i,vectors):
            dd = distf(x[i,:], x[j,:])
            d[i,j] = dd
            d[j,i] = dd

    return d


def rank_matrix(x):
    """Returns rank matrix from pairwise distance matrix a"""
    
    m = x.argsort()
    r = numpy.zeros(x.shape)

    vectors = x.shape[0]

    for i in range(vectors):
        for j in range(vectors):
            pos = numpy.where(m[i,:]==j)
            r[i,j] = pos[0][0] # there should be a better syntax for this
        
    return r.astype('int')


def centering(x):
    """Center matrix x to origo"""
    return x - x.mean(axis=0)


def double_centering(x):
    """Double center matrix x"""
    pass

def trustworthiness(orig, proj, ks):
    """Calculate a trustworthiness values for dataset.

    orig
      matrix containing the data in the original space

    proj
      matrix containing the data in the projected space

    ks range indicating neighbourhood(s) for which
      trustworthiness is calculated.

    Return list of trustworthiness values

    """

    dd_orig = distance_matrix(orig)
    dd_proj = distance_matrix(proj)
    nn_orig = dd_orig.argsort()
    nn_proj = dd_proj.argsort()

    ranks_orig = rank_matrix(dd_orig)

    trust = []
    for k in ks:
        moved = []
        for i in range(orig.shape[0]):
            moved.append(moved_in(nn_orig, nn_proj, i, k))

        trust.append(trustcont_sum(moved, ranks_orig, k))

    return trust

def continuity(orig, proj, ks):
    """Calculate a continuity values for dataset

    orig
      matrix containing the data in the original space

    proj
      matrix containing the data in the projected space

    ks range indicating neighbourhood(s) for which continuity
      is calculated.

    Return a list of continuity values
    """

    dd_orig = distance_matrix(orig)
    dd_proj = distance_matrix(proj)
    nn_orig = dd_orig.argsort()
    nn_proj = dd_proj.argsort()

    ranks_proj = rank_matrix(dd_proj)

    cont = []
    for k in ks:
        moved = []
        for i in range(orig.shape[0]):
            moved.append(moved_out(nn_orig, nn_proj, i, k))

        cont.append(trustcont_sum(moved, ranks_proj, k))

    return cont

def moved_out(nn_orig, nn_proj, i, k):
    """Determine points that were neighbours in the original space,
    but are not neighbours in the projection space.

    nn_orig
      neighbourhood matrix for original data

    nn_proj
      neighbourhood matrix for projection data

    i
      index of the point considered

    k
      size of the neighbourhood considered

    Return a list of indices for 'moved out' values 
    """

    oo = list(nn_orig[i, 1:k+1])
    pp = list(nn_proj[i, 1:k+1])

    for j in pp:
        if (j in pp) and (j in oo):
            oo.remove(j)

    return oo

def moved_in(nn_orig, nn_proj, i, k):
    """Determine points that are neighbours in the projection space,
    but were not neighbours in the original space.

    nn_orig
      neighbourhood matrix for original data

    nn_proj
      neighbourhood matrix for projection data

    i
      index of the point considered

    k
      size of the neighbourhood considered

    Return a list of indices for points which are 'moved in' to point i
    """

    pp = list(nn_proj[i, 1:k+1])
    oo = list(nn_orig[i, 1:k+1])

    for j in oo:
        if (j in oo) and (j in pp):
            pp.remove(j)

    return pp


def scaling_term(k, n):
    """Term that scales measure between zero and one

    k  size of the neighbourhood
    n  number of datapoints
    """
    if k < (n / 2.0):
        return 2.0 / ((n*k)*(2*n - 3*k - 1))
    else:
        return 2.0 / (n * (n - k) * (n - k - 1))


def trustcont_sum(moved, ranks, k):
    """Calculate sum used in trustworthiness or continuity calculation.

    moved

       List of lists of indices for those datapoints that have either
       moved away in (Continuity) or moved in (Trustworthiness)
       projection

    ranks

       Rank matrix of data set. For trustworthiness, ranking is in the
       original space, for continuity, ranking is in the projected
       space.

    k
       size of the neighbournood
    """

    n = ranks.shape[0]
    s = 0

    # todo: weavefy this for speed
    for i in range(n):
        for j in moved[i]:
            s = s + (ranks[i, j] - k)

    a = scaling_term(k, n)

    return 1 - a * s