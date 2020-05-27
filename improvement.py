"""Functions that perform node clustering iteratively.

The functions in this file are 'iterative' functions, meaning that on a run
you start by an initial clustering and you keep improving it using an iterative method.
There are intermediate clusterings in the process of finding the best one.
(To contrast with the 'one way' methods in the clustering.py )

There are two categories based on the goal of the algorithms. The first is the
max intra / min inter category where the objective is to minimize the inter cluster
traffic and maximize the intra cluster traffic. The second category has as a goal
to minimize the maximum inter cluster flow.
"""
import operator
from math import exp, factorial

from nodeflows import *
from clustering import *
from constants import MINIMUM_NETWORK_SIZE

'''
----------------- Max intra / Min inter ------------------------------------------------------------------------
Goal of these methods is to Maximize the intra cluster communication / Minimize the inter cluster communication.
'''


def stochastic_best_candidates(clusters, node_flows, parts=1):
    """Improves the network nodes clustering using the stochastic best candidates technique.

    Finds X pairs that are heuristically good swapping candidates and calculates a probability
    of swapping. Then using those probabilities the swapping pair is picked and the swap is
    performed. If there are no improving swaps then the process ends.

    Args:
        clusters (Cluster): the initial clustering
        node_flows (NodeFlows): the node flows metrics
        parts (int): the number switches in the cluster membership structure (default value: 1)

    Returns:
       tuple: a tuple of the form
            (new clusters, new node flows, number of iteration)
    """

    '''
    Basic pseudocode:

    state = <initial value>;
    while state not optimal:
        Find some candidate new states
        Calculate probability of improvement
        Stochasticaly pick one of them, s
        state = s
        evaluate(state)
    '''

    # initialize search
    nf = node_flows
    iteration = 0

    # Intuition:
    # Number of candidates: for a network instance G with  N nodes and cluster size d
    # we have N*(N-d)/2 possible pairs from which we can use to perform a swap and move
    # to the next network state. Now, the number X of candidates per iteration should
    # be analogous to the network size, yet the bigger, the more the calculations.
    # A valid choice would be to have at least as many candidate pairs as the existing clusters
    # plus a constant which is picked - randomly - to be 2 for small networks and 10 for the bigger ones

    plus = 10
    if clusters.N < MINIMUM_NETWORK_SIZE:
        plus = 2
    num_of_candidates = clusters.M + plus

    net_states = {}

    pairs = nf.get_pairs(num_of_candidates, clusters, parts)
    intra = nf.intra

    # stop if none of the candidates leads to better solution
    while (list(pairs.values()) > intra).sum() > 0:

        # expected to enhance bigger value distance from the current state
        vals = list(exp((x - intra) / intra) for x in pairs.values())
        tot = sum(vals)
        probs = np.array(vals) / tot

        # find next swap pair probabilistically
        idx = np.random.choice(range(len(pairs)), p=probs)
        n1, n2 = list(pairs.keys())[idx]

        clusters.swap(n1, n2)
        nf.update((n1, n2), clusters)
        intra = nf.intra

        key = tuple(clusters.lookup)
        if key not in net_states:
            net_states[key] = intra

        pairs = nf.get_pairs(num_of_candidates, clusters, parts)
        iteration += 1

    if len(net_states.keys()) > 0:
        lookup = max(net_states.items(), key=operator.itemgetter(1))[0]
        new_clusters = Clusters(clusters.N, clusters.M, clusters.d, list(lookup))
        return new_clusters, NodeFlows.calculate_flows(new_clusters, nf.T), iteration
    else:
        return clusters, nf, iteration


def iterative_random_swaps(clusters, node_flows, parts=1):
    """Improves the network nodes clustering using random nodes swapping.

     Picks randomly pairs of nodes. If the pair results into an improvement then
     the swap is performed. The process is repeated until not X swaps at least
     improve the clustering, where X is size related heuristic value.

     Args:
         clusters (Cluster): the initial clustering
         node_flows (NodeFlows): the node flows metrics
         parts (int): the number switches in the cluster membership structure (default value: 1)

     Returns:
        tuple: a tuple of the form
             (new clusters, new node flows, number of iteration)
     """

    '''
    Basic pseudocode:

    state = <initial value>;
    while state not optimal:
        Find a better state s
        state = s
        evaluate(state)
    '''

    def next_pair(clusters, parts=1):
        """Finds a valid swapping pair with respect to the number of cluster membership switches.

         Args:
             clusters (Cluster): the initial clustering
             parts (int): the number switches in the cluster membership structure (default value: 1)

         Returns:
            tuple: a pair of nodes to swap
         """

        # case 1 : both on the same partition -> the swap can be performed
        # case 2 : if in different partitions then the swap cannot be performed
        # partitions are PART1: [1-...N/parts] PART2: [N/parts+1....2*N/parts] etc. and they have same size

        n1 = random.choice(list(range(clusters.N)))
        # identify part in which node belongs
        part_size = N // parts
        origin_part = (n1 // part_size)*part_size
        valid_choices = set(range(origin_part, origin_part + part_size))
        n1cluster = clusters.get_nodes(clusters.cluster_of(n1))
        to_choose_from = valid_choices - set(n1cluster)
        n2 = random.choice(list(to_choose_from))

        return n1, n2

    N = clusters.N
    d = clusters.d
    M = clusters.M
    part_size = N//parts
    portion_for_cluster = part_size//M

    available_swaps = N * (part_size - portion_for_cluster) // 2

    # TODO (gina): this is intuition based
    baseline = 10
    # net_states_num = factorial(N)/factorial(M) #roughly
    max_iterations = baseline + log(factorial(N), 10 if N >= 64 else 2) \
                     - log(factorial(M), 10 if N >= 64 else 2)

    # init stopping mechanism
    tried_to_swap = set()
    net_states = {}

    iteration = 0
    maximum_state_returns = 3

    percent = 0.2
    thres = available_swaps * (1 - percent)

    nf = node_flows
    while len(tried_to_swap) <= thres:
        n1, n2 = next_pair(clusters, parts)

        if nf.get_value((n1, n2), clusters) >= nf.intra:

            tried_to_swap.clear()
            clusters.swap(n1, n2)
            nf.update((n1, n2), clusters)
            iteration += 1

            # TODO (gina): take it out if very slow
            ord = clusters.get_ordinal()
            net_states[str(ord)] = net_states.get(str(ord), 0) + 1
            if net_states[str(ord)] == maximum_state_returns:
                return clusters, nf, iteration
        else:
            tried_to_swap.add((n1, n2))

    return clusters, nf, iteration


'''
----------------- Min max inter ---------------------------------------------------------------------------------

Basic idea of these methods is that by minimizing the maximum inter cluster flow we can create a clustering 
with minimum inter cluster traffic and balanced inter cluster flows. Also by enforcing this in the inter cluster
flows in a way we request the bigger flows to be inside the clusters and not occupy the cluster network links.
'''

# TODO (gina): Complete this part
