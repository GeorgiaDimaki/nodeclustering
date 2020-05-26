"""Functions for traffic matrix preprocessing.

In many situations it is beneficial to run clustering algorithms on sparser or
differently structured traffic matrices. This file contains a set of possible
traffic matrix transformations to apply prior to a clustering algorithm.
"""

import numpy as np


def maxperce(T):
    """Normalizes the traffic matrix based on the max flow value.

     Args:
        T (list): the traffic matrix.

    Returns:
        numpy.ndarray: the normalized traffic matrix
    """
    t = np.array(T)
    return t/max(t.flatten())


def discard(T, limit):
    """ Discards matrix values (flows) that are smaller than the limit.

    Args:
        T (list): the traffic matrix.
        limit (float): the smaller value that can exist on the traffic matrix.
    Returns:
        numpy.ndarray: the traffic matrix after the smaller values discard.
    """
    T = np.array(T)
    T[T <= limit] = 0
    return T


def normalize(T):
    """Normalizes the traffic matrix based on the total sum of flows on the matrix.

    In certain literature this matrix represents the 'shape' of the traffic.

     Args:
        T (list): the traffic matrix.

    Returns:
        numpy.ndarray: the normalized traffic matrix
    """
    return np.array(T)/sum(T)


def similarity_matrix(T):
    """Creates the similarity matrix of the network nodes.

    The similarity in the case of network nodes can be assumed as the mutual traffic every pair of
    nodes exchanges. Thus this creates the mutual traffic symmetric matrix, where the values
    can be thought as the similarity of the network nodes.

     Args:
        T (list): the traffic matrix.

    Returns:
        numpy.ndarray: the similarity traffic matrix.
    """

    adj = np.array(T)
    adj = adj + np.transpose(adj)

    return adj
