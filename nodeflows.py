import numpy as np
import bottleneck as bn

# TODO (gina): Integrate Node flows with the clusters. A node flow object can contain the clustering.
#              can be seen as a network object in general with clustering and flows knowledge


class NodeFlows(object):
    """Tracks and manages the flows a node exchanges with each cluster.

    Attributes:
        T (numpy.ndarray): the NxN traffic matrix
        fto (numpy.ndarray): the NxM traffic matrix of flows from node i to cluster j
        ffrom (numpy.ndarray): the NxM traffic matrix of flows from cluster j to node i
        intra (float): the intra cluster traffic
        ratio (numpy.ndarray): NxM matrix of ratios of the form
            traffic exchanged with node i and cluster j / traffic i exchanges with its own cluster
        ratios_used (bool): Whether or not the algorithm uses the ratios

    """

    def __init__(self, T, fto, ffrom, intra, ratio=None, ratios_used=False):
        """Initializes a NodeFlows object.

        Args:
            T (numpy.ndarray): the NxN traffic matrix
            fto (numpy.ndarray): the NxM traffic matrix of flows from node i to cluster j
            ffrom (numpy.ndarray): the NxM traffic matrix of flows from cluster j to node i
            intra (float): the intra cluster traffic
            ratio (numpy.ndarray): NxM matrix of ratios of the form
                traffic exchanged with node i and cluster j / traffic i exchanges with its own cluster
                (default value: None)
            ratios_used (bool): Whether or not the algorithm uses the ratios (default value: False)

        Returns:
            no value
        """
        self.T = T
        self.f_to = fto
        self.f_from = ffrom
        self.ratio = ratio
        self.intra = intra
        self.ratios_used = ratios_used

    def __copy__(self):
        """Copies the NodeFlow instance.

        Returns:
           NodeFlows: a copy of the current instance
        """
        return NodeFlows(self.T, self.f_to, self.f_from, self.ratio, self.intra)

    @staticmethod
    def calculate_flows(clusters, T, ratios_used=False):
        """Calculates the NodeFlows object of a given clustering over a matrix T.

        Args:
            clusters (Clusters): the clusters
            T (numpy.ndarray): the traffic matrix
            ratios_used (bool): Whether or not the algorithm uses the ratios

        Returns:
            NodeFlows: the calculated node flows object
        """
        ratios = None
        traffic = np.array(T)
        mem = clusters.get_membership_matrix()
        f_from = mem @ traffic
        f_from = np.transpose(f_from)
        f_to = traffic @ np.transpose(mem)

        own = np.zeros(clusters.N)
        for n in range(clusters.N):
            j = clusters.cluster_of(n)
            own[n] = f_from[n, j] + f_to[n, j]
        intra = sum(own)

        if ratios_used:
            ratios = f_to + f_from
            ratios = ratios / own
            for n in range(clusters.N):
                j = clusters.cluster_of(n)
                ratios[n, j] = 0

        return NodeFlows(T, f_to, f_from, intra, ratios, ratios_used)

    # TODO (gina): Optimize that
    def update_ratios(self, pair, clusters):
        """Updates the ratios matrix.

        Args:
            pair (tuple): a pair of nodes that got swapped
            clusters (Clusters): the clusters

        Returns:
            no value
        """
        i, j = pair
        u = clusters.cluster_of(i)  # i moved from v to u
        v = clusters.cluster_of(j)  # j moved from u to v
        self.ratio = np.zeros(shape=(clusters.N, clusters.M))
        for n in range(clusters.N):
            j = clusters.cluster_of(n)
            own = self.f_to[n, j] + self.f_from[n, j]
            for u in clusters.get_names():
                if u == j:
                    continue
                self.ratio[n, u] = (self.f_to[n, u] + self.f_from[n, u]) / own

    def update_flows(self, pair, clusters):
        """Updates the flows.

        Args:
            pair (tuple): a pair of nodes that got swapped
            clusters (Clusters): the clusters

        Returns:
            no value
        """
        i, j = pair
        u = clusters.cluster_of(i)  # i moved from v to u
        v = clusters.cluster_of(j)  # j moved from u to v

        for n in range(len(self.T[0])):
            # traffic exchanged with cluster u
            self.f_to[n, u] += self.T[n][i] - self.T[n][j]
            self.f_from[n, u] += self.T[i][n] - self.T[j][n]

            # traffic exchanged with culster v
            self.f_to[n, v] += self.T[n][j] - self.T[n][i]
            self.f_from[n, v] += self.T[j][n] - self.T[i][n]

        i_change = self.f_to[i, u] + self.f_from[i, u] - \
                   (self.f_to[i, v] - self.T[i][j]) - \
                   (self.f_from[i, v] - self.T[j][i])

        j_change = self.f_to[j, v] + self.f_from[j, v] - \
                   (self.f_to[j, u] - self.T[j][i]) - \
                   (self.f_from[j, u] - self.T[i][j])

        self.intra += i_change + j_change

    def update(self, pair, clusters):
        """Updates NodeFlows instance after a swap.

        To be called on updated clusters (After swap)

        Args:
            pair (tuple): a pair of nodes that got swapped
            clusters (Clusters): the clusters

        Returns:
            bool: whether the update happened correctly
        """
        self.update_flows(pair, clusters)
        if self.ratios_used:
            self.update_ratios(pair, clusters)

        return True

    def get_value(self, pair, clusters):
        """Returns the objective value after the swap of the pair is performed.

        Args:
            pair (tuple): the pair of nodes to swap
            clusters (Clusters): the clusters

        Returns:
            float: the objective value after the swap
        """
        i, j = pair
        u = clusters.cluster_of(i)  # i moved from u to v
        v = clusters.cluster_of(j)  # j moved from v to u

        # traffic from i to v apart from Tij is in the intra now
        # traffic from v to i apart from Tji is in the intra now
        # traffic from i to u is not in the intra now
        # traffic from u to i is not in the intra now
        i_change = (self.f_to[i, v] - self.T[i][j]) + \
                   (self.f_from[i, v] - self.T[j][i]) - \
                   self.f_to[i, u] - \
                   self.f_from[i, u]

        # traffic from j to u apart from Tji is in the intra now
        # traffic from u to j apart from Tij is in the intra now
        # traffic from j to v is not in the intra now
        # traffic from j to v is not in the intra now
        j_change = (self.f_to[j, u] - self.T[j][i]) + \
                   (self.f_from[j, u] - self.T[i][j]) - \
                   self.f_to[j, v] - \
                   self.f_from[j, v]

        return self.intra + i_change + j_change

    def get_pairs(self, num, clusters, parts=1):
        """Returns <num> candidate pairs to exchange.

        It picks <num> nodes with the highest ratios (the highest the value the more the traffic
        a node exchanges with a cluster other that its own), finds the destination clusters
        and from there picks the node each node is better to swap with (based on their respective
        ratios as well)

        Args:
            num (int): number of candidate pairs to swap
            clusters (Clusters): the clusters
            parts (int): the number of switches in the cluster membership structure

        Returns:
            dict: a dictionary of the form { <pair> : <objective value after the swap> }
        """

        index = bn.argpartition(-self.ratio.flatten(), num - 1)
        M = len(self.f_to[0])
        N = len(self.f_to)
        part_size = N//parts

        pairs = {}
        for i in range(num):
            node = index[i] // M
            origin_part = node // part_size

            mask = np.zeros(shape=clusters.N)
            mask[origin_part*part_size: (origin_part+1)*part_size] = 1

            cluster = index[i] % M
            nodecluster = clusters.cluster_of(node)

            if cluster == nodecluster:
                print(self.ratio.flatten()[index[i]])
            indic = np.array(clusters.get(cluster))

            candidates = indic * self.ratio[:, nodecluster] * mask
            node2 = np.argmax(candidates)
            pairs[(node, node2)] = self.get_value((node, node2), clusters)
        return pairs
