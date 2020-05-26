class Clusters:
    """Represents and manages the nodes clusters.

    Attributes:
        clusters (dict): a dictionary of the form
                    {cluster_num : {'nodes': <the node set of the cluster>,
                                   'indicator': <the indicator 0-1 vector of nodes participation in the cluster>}}

        lookup (list): a cluster membership lookup table. Place i contains the cluster of node i.
        N (int): th number of network nodes
        M (int): the number of clusters
        d (int): the number of nodes per cluster
    """

    def __init__(self, N, M, d, lookup=()):
        """Initializes a clusters object.

        Args:
            N (int): th number of network nodes
            M (int): the number of clusters
            d (int): the number of nodes per cluster
            lookup (list): a cluster membership lookup table.
                           Place i contains the cluster of node i. (default value: ())

        Returns:
           str: the path to the log files
        """
        self.clusters = dict((i, {'nodes': set(), 'indicator': [0 for i in range(N)]}) for i in range(M))
        self.lookup = lookup
        if len(self.lookup) == 0:
            self.lookup = [None for i in range(N)]
        else:
            for i in range(N):
                self.assign(i, lookup[i], False)

        self.N = N
        self.d = d
        self.M = M

    def assign(self, node, cluster, update_lookup=True):
        """Assigns a node to a cluster.

        Args:
            node (int): the node to be assigned into a cluster
            cluster (int): the cluster were the node gets assigned
            update_lookup (bool): whether to update the lookup table or not

        Returns:
            no value
        """

        self.clusters[cluster]['nodes'].add(node)
        self.clusters[cluster]['indicator'][node] = 1

        if update_lookup or self.lookup[node] is None:
            self.lookup[node] = cluster

    def is_full(self, cluster):
        """Returns whether the cluster is full or not.

        Args:
            cluster (int): the cluster to check

        Returns:
           bool: True if the cluster is full
        """
        return sum(self.clusters[cluster]['indicator']) == self.d

    def is_oversized(self, cluster):
        """Returns whether the cluster has more nodes than it should.

        Args:
            cluster (int): the cluster to check

        Returns:
           bool: True if the cluster is bigger than it should
        """
        return sum(self.clusters[cluster]['indicator']) > self.d

    def is_undersized(self, cluster):
        """Returns whether the cluster has less nodes than it should.

        Args:
            cluster (int): the cluster to check

        Returns:
           bool: True if the cluster is smaller than it should
        """
        return sum(self.clusters[cluster]['indicator']) < self.d

    def get(self, cluster):
        """Returns the membership indicator vector of the cluster.

        Args:
            cluster (int): the cluster to check

        Returns:
           list: the membership indicator vector
        """
        return self.clusters[cluster]['indicator']

    def get_membership_matrix(self):
        """Returns a two dimensional array that for rows has the membership indicator of each cluster.

        Returns:
           numpy.ndarray: the MxN membership indicators array
        """
        import numpy as np
        matrix = []
        for i in self.clusters:
            matrix.append(self.clusters[i]['indicator'])
        matrix = np.array(matrix)
        return matrix

    def get_nodes(self, cluster):
        """Returns the nodes of a cluster.

        Args:
            cluster (int): the cluster of interest

        Returns:
           list: the cluster's nodes
        """
        return list(self.clusters[cluster]['nodes'])

    def get_instance(self):
        """Returns the full clusters instance.

        Returns:
           dict: the clusters
        """
        return self.clusters

    def get_names(self):
        """Returns the names of the clusters (1, 2, .., M).

        Returns:
           list: the clusters names
        """
        return self.clusters.keys()

    def remove(self, node, cluster):
        """Removes a node from a cluster.

        We expect that is a temporary removal. The lookup table value is -1, the same for unassigned nodes.

        Args:
            node (int): the node to remove
            cluster (int): the cluster to remove the node from

        Returns:
            no value
        """
        self.clusters[cluster]['indicator'][node] = 0
        self.clusters[cluster]['nodes'].remove(node)
        self.lookup[node] = -1

    def add(self, node, cluster):
        """Adds a node to a cluster.

        The same as assign but in a list notation. It always updates the lookup

        Args:
            node (int): the node to add
            cluster (int): the cluster of interest

        Returns:
            no value
        """
        self.assign(node, cluster, True)

    def move(self, node, to_c, from_c=None):
        """Moves a node from its cluster to a new cluster.

        Args:
            node (int): the node to move
            to_c (int): the destination cluster
            from_c (int): the origin cluster

        Returns:
            no value
        """
        if not from_c:
            from_c = self.lookup[node]
        self.remove(node, from_c)
        self.add(node, to_c)

    def cluster_of(self, node):
        """Returns the cluster of the node.

        Args:
            node (int): the node whose cluster to find

        Returns:
            int: the cluster
        """
        return self.lookup[node]

    def is_member_of(self, node, cluster):
        """Returns whether or not the node belongs to the cluster.

        Args:
            node (int): the node to check
            cluster (int): the cluster of interest

        Returns:
            bool: True if the node belongs to the clustr
        """
        return self.lookup[node] == cluster

    def are_neighbours(self, n1, n2):
        """Returns whether or not two nodes belong in the same cluster aka they are neighbors.

        Args:
            n1 (int): node 1
            n2 (int): node 2

        Returns:
            bool: True if the nodes are in the same cluster
        """
        return self.lookup[n1] == self.lookup[n2]

    def swap(self, n1, n2):
        """Exchanges the clusters of node 1 and node 2.

        Args:
            n1 (int): node 1
            n2 (int): node 2

        Returns:
            no value
        """
        c1 = self.cluster_of(n1)
        c2 = self.cluster_of(n2)

        self.move(n1, c2, c1)
        self.move(n2, c1, c2)

    def get_ordinal(self):
        """Gets a unique representation of the nodes assortments (clusters).

        We can give a unique representation of the assortments if we order the clusters
        based on their smallest element and naming them using this ordering.

        Returns:
            list: the new unique lookup table
        """
        track = {}
        optrack = {}
        seen = set()
        node = 0
        while len(seen) < self.M:

            '''
            Since lookup is indexed with nodes in order, the first time I read
            one of the clusters identifier is by the smallest node in this cluster.

            First I find the smallest node in each cluster's identifier

            Then, I want to order my sets in terms of their smallest element
            and assign to the set the corresponding order identifier.
            Since I read linearly my nodes, I immediately know the ordering.
            
            Example: 
                if node 5 is the first one with identifier "6" and I know that
                I already had 3 identifiers "seen", then I know that in the new ordering
                the set that includes 5 should have identifier "3" (instead of "6")

            optrack is just another dictionary to help us fast make the new assignment
            The full mapping happens in time O(N).
            '''

            if self.lookup[node] not in seen:

                seen.add(self.lookup[node])
                track[node] = len(seen) - 1
                optrack[self.lookup[node]] = node
            node += 1

        new_lookup = [0]*self.N
        for i in range(self.N):
            new_lookup[i] = track[optrack[self.lookup[i]]]

        return new_lookup
