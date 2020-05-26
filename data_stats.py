"""Functions that calculate useful for the algorithms information on the traffic matrices.
"""


def get_stats(T, type_id, N=64, M=2, d=32):
    """Calculates the traffic flows in intra and inter cluster communication of the true clusters.

    This function is basically useful for 'engineered' traffic matrices for which we know the
    true clusters. It calculates the actual traffic flows in intra and inter cluster communication
    that the random matrix represents. Needed for evaluation of the algorithms' performance.

    Args:
        T (numpy.ndarray): the traffic matrix
        type_id (int): the traffic type code, one of {0,3,5,6,7,8}
        N (int): the number of network nodes
        M (int): the number of clusters
        d (int): the number of nodes per cluster

    Returns:
        tuple: a tuple of the form <total traffic exchanges, intra cluster traffic, inter cluster traffic>
    """

    tot = sum(sum(T))
    inter = 0

    if type_id == 0:
        # unit out clustered
        inter = sum(x for x in T.flatten() if x == 1)
    elif type_id == 3:
        # uniformly clustered
        inter = 0
    elif type_id == 5:
        # controlled
        inter = sum(x for x in T.flatten() if x < 7)
    elif type_id == 6:
        # controlled dense
        inter = sum(x for x in T.flatten() if x < 7)
    elif type_id == 7:
        # unit all to all
        # every node sends a unit of traffic to every node outside its cluster
        inter = N * (N - d)
    elif type_id == 8:
        # dense_clustered
        inter = sum(x for x in T.flatten() if x < 7)

    intra = tot - inter

    return tot, intra, inter
