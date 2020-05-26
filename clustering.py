"""Functions that perform node clustering.

The functions in this file are 'one way' functions, meaning that on a run
they create a clustering and there are no intermediate clustering.
(To contrast with the iterative methods in the improvement.py )
"""

from sklearn.cluster import SpectralClustering
import random
from preprocessing import *
from math import log
from clusters import *


# TODO (gina): rename to random_clustering
def slow_clustering(T, M, parts=1):
    """Randomly clusters the network nodes.

    Args:
        T (list): the traffic matrix
        M (int): the number of clusters
        parts (int): the number switches in the cluster membership structure (default value: 1)

    Returns:
       Clusters: the clustering of nodes
    """
    d = len(T[0]) // M
    portion = d // parts
    part_size = len(T[0]) // parts
    clusters = Clusters(len(T), M, d, [])

    for part in range(parts):
        nodes = [i for i in range(part * part_size, (part + 1) * part_size)]
        for c in range(M):
            for _ in range(portion):
                n = random.choice(nodes)
                clusters.assign(n, c, True)
                nodes.remove(n)

    return clusters


# TODO (gina): remove. Keep only spectral clustering function
def fast_clustering(T, M, parts=1):

    return spectral_clustering(T, M)


def spectral_clustering(T, M, parts=1):
    """Clusters the network nodes using Spectral clustering.

    Given a traffic matrix T and the number of clusters M it uses spectral clustering
    to assign nodes into clusters. If the clusters are not of equal size - sklearn.SpectralClustering
    does not guarantee they will - a balancing technique is used to make them similarly sized.

    Args:
        T (list): the traffic matrix
        M (int): the number of clusters
        parts (int): the number switches in the cluster membership structure (default value: 1)

    Returns:
       Clusters: the clustering of nodes
    """
    # mutual traffic
    adj = similarity_matrix(T)

    # post mutual traffic discarding
    adj = maxperce(adj)
    adj = discard(adj, 0.5)

    sc = SpectralClustering(M, affinity='precomputed',
                            eigen_solver='amg',
                            assign_labels='discretize',
                            random_state=1022, n_neighbors=round(log(len(T))), n_jobs=-1)

    sc.fit(adj)
    d = len(T[0]) // M
    clusters = Clusters(len(T), M, d, sc.labels_)
    for i in range(len(sc.labels_)):
        clusters.assign(i, sc.labels_[i], False)

    for i in range(M):
        if clusters.is_oversized(i):
            return balance_clusters(clusters, d, T)

    return clusters


def balance_clusters(clusters, d, T):
    """Creates equal sized clusters when the clustering is unbalanced.

    Greedily picks nodes from an oversized clusters and assigns them to undersized clusters.

    Args:
        clusters (Clusters): the clusters
        d (int): the number of nodes per clusters
        T (list): the traffic matrix

    Returns:
       Clusters: the clustering of nodes in equal sized clusters
    """
    c = set(clusters.get_names())
    small = dict()
    big = set()

    # If in the set there are not correctly sized clusters then I must have both small and big clusters
    for i in c:
        if clusters.is_undersized(i):
            small[i] = clusters.get(i)
        elif clusters.is_oversized(i):
            big.add(i)

    traffic = similarity_matrix(T)
    for b in big:
        current = np.array(clusters.get(b))
        while sum(current) - d > 0:

            # find the cluster's node that has the min max flow to its neighbors
            min_max = np.inf
            to_change = 0
            for i in range(len(current)):
                if current[i]:
                    m = np.max(np.multiply(traffic[i, :], current))
                    if m < min_max:
                        min_max = m
                        to_change = i

            # find the small cluster it exchanges most traffic with
            to = list(small.keys())
            values = []
            for s in to:
                values.append(traffic[i, :] @ small[s])

            best = to[np.argmin(np.array(values))]

            clusters.move(to_change, best, b)
            small[best][to_change] = 1
            current[to_change] = 0

            if clusters.is_full(best):
                small.pop(best)

    return clusters


def greedy_clustering(T, M, parts=1):
    """Clusters the network nodes using greedy pairing.

    Orders the pairs of nodes based on their mutual traffic and rotationally assigns them into
    clusters.

    Args:
        T (list): the traffic matrix
        M (int): the number of clusters
        parts (int): the number switches in the cluster membership structure (default value: 1)

    Returns:
       Clusters: the clustering of nodes
    """

    N = len(T[0])
    d = N // M
    part_size = N // parts
    portion = d // parts

    nodes = set(list(range(len(T))))
    lookup = [-1] * len(T)

    # accounting
    no_space_for_pair_clusters = set()
    portions_size = {i: {'csize':{c:0 for c in range(M)},
                         'no_space_for_pair_clusters': set(range(M)) if d == parts else set() } \
                     for i in range(parts)}
    clusters_size = {i:0 for i in range(M)}

    # mutual traffic
    adj = np.array(T, dtype=np.float64)
    adj = adj + np.transpose(adj)

    # short in descending order
    adj = -np.tril(adj)
    indices = np.unravel_index(np.argsort(adj, axis=None), adj.shape)

    # Notes: If portion within the partition is full then no more nodes can be assigned from that partition

    current_c = 0
    index = -1
    while len(nodes) > 0 and index + 1 < len(indices[0]):
        # prepare for the next round
        index += 1
        i = indices[0][index]
        j = indices[1][index]

        # case 1: both nodes unassigned. We put them in the same cluster rotating through clusters
        # case 2: both nodes assigned!. We skip the pair
        # case 3: One of the nodes belongs to a cluster.
        #           if cluster full then we skip the pair
        #           if cluster not full then we assign the other node there

        c1 = lookup[i]
        c2 = lookup[j]

        # cluster check: is there space for pair of one node?
        # partition check: do the nodes belong to the same partition?
        partition_i = i // part_size
        partition_j = j // part_size
        same_partition = partition_j == partition_i

        # portion check: is there space for pair or node?

        if c1 == c2:    # they had the same assignment

            # Cases they cannot be assigned together:
            # 1: They are both assigned already in the same cluster --> the if is excluding those pairs

            if c1 == -1:

                # 2: There is no space for a pair in any cluster
                #    --> I have to start considering individual nodes
                if len(no_space_for_pair_clusters) == M:
                    continue

                # 3: They are in the same partition and there is no space for a pair in the portion

                if same_partition:

                    # no cluster has space for a pair  in this partition
                    if len(portions_size[partition_i]['no_space_for_pair_clusters']) == M:
                        continue

                    # first  move to the next cluster in the cycle
                    current_c = (current_c + 1) % M
                    # if not stopped before there must be one cluster with space for a pair
                    while portions_size[partition_i]['csize'][current_c] > portion - 2:
                        current_c = (current_c + 1) % M

                else:

                    # 4: They are not in the same partition and there is no space in the same cluster in their
                    #    respective partitions.
                    clusterstamp = current_c
                    cycled = False
                    while portions_size[partition_i]['csize'][current_c] == portion or \
                        portions_size[partition_j]['csize'][current_c] == portion:
                        current_c = (current_c + 1) % M
                        if clusterstamp == current_c:
                            cycled = True
                            break

                    if cycled:
                        current_c = (current_c + 1) % M
                        continue

                # complete the assignment
                lookup[i] = current_c
                lookup[j] = current_c
                nodes.remove(i)
                nodes.remove(j)

                # update all the counters
                clusters_size[current_c] += 2
                if clusters_size[current_c] > d-2:
                    no_space_for_pair_clusters.add(current_c)

                portions_size[partition_i]['csize'][current_c] += 2 if same_partition else 1
                if portions_size[partition_i]['csize'][current_c] > portion - 2:
                    portions_size[partition_i]['no_space_for_pair_clusters'].add(current_c)

                if not same_partition:
                    portions_size[partition_j]['csize'][current_c] += 1
                    if portions_size[partition_j]['csize'][current_c] > portion - 2:
                        portions_size[partition_j]['no_space_for_pair_clusters'].add(current_c)
            # else they are in the same cluster already and we should continue
        else:
            # This means that either both assigned or the one unassigned (lookup entry -1)

            # Case they cannot be assigned in the same cluster:
            # 1: They are both assigned already ---> taken care of by the if
            # 2: One is assigned but there is no space in this cluster for a new node
            # 3: One is assigned but there is no space in this cluster within the partition for a new node

            if c1 == -1:
                # Case 2
                if clusters_size[c2] == d:
                    continue

                # Case 3
                if portions_size[partition_i]['csize'][c2] == portion:
                    continue

                lookup[i] = c2
                nodes.remove(i)

                # update all the counter
                clusters_size[c2] += 1
                if clusters_size[c2] > d - 2:
                    no_space_for_pair_clusters.add(c2)

                portions_size[partition_i]['csize'][c2] += 1
                if portions_size[partition_i]['csize'][c2] > portion - 2:
                    portions_size[partition_i]['no_space_for_pair_clusters'].add(c2)

            if c2 == -1:
                # Case 2
                if clusters_size[c1] == d:
                    continue

                # Case 3
                if portions_size[partition_j]['csize'][c1] == portion:
                    continue

                lookup[j] = c1
                nodes.remove(j)

                # update all the counter
                clusters_size[c1] += 1
                if clusters_size[c1] > d - 2:
                    no_space_for_pair_clusters.add(c1)

                portions_size[partition_j]['csize'][c1] += 1
                if portions_size[partition_j]['csize'][c1] > portion - 2:
                    portions_size[partition_j]['no_space_for_pair_clusters'].add(c1)

    return Clusters(len(T), M, d, lookup)


def nn_clustering(T, M, parts=1):
    """Clusters the network nodes using the nearest neighbors.

    Orders the nodes with descending order of the variance of their mutual traffic. Then the
    node with the highest variance picks d-1 nodes to pair up with in a cluster. The
    process is repeated until al nodes have a cluster.

    Args:
        T (list): the traffic matrix
        M (int): the number of clusters
        parts (int): the number switches in the cluster membership structure (default value: 1)

    Returns:
       Clusters: the clustering of nodes
    """

    # When the node has to decide on neighbors he has to pick a proportion from each partition

    d = len(T[0]) // M
    part_size = len(T[0])//parts
    portion = d // parts

    lookup = [M-1] * len(T)
    # mutual traffic
    adj = np.array(T)
    adj = adj + np.transpose(adj)

    nodes = np.array(list(range(len(T[0]))))

    for c in range(M - 1):
        # short in descending order
        var_tr = np.var(adj, axis=0)
        idx = np.argpartition(-var_tr, 1)[:1][0]  # finds the index of the maximum
        origin_part = idx // part_size

        current_cl = np.array([idx])
        for p in range(parts):
            neis = portion -1 if p == origin_part else portion

            current_cl = np.append(current_cl,
                                   p*part_size + np.argpartition(-adj[idx, p*part_size : (p+1)*part_size], neis)[:neis])

        for i in current_cl:
            lookup[nodes[i]] = c

        adj = np.delete(adj, current_cl, axis=0)
        adj = np.delete(adj, current_cl, axis=1)
        nodes = np.delete(nodes, current_cl, axis=0)

        part_size -= portion

    return Clusters(len(T), M, d, lookup)

