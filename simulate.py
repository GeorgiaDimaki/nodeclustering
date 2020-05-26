from time_calc import *
from logger import *
from math import ceil
from constants import *
from data_stats import get_stats
from os import path, getcwd, mkdir, environ
from improvement import *
from results import *


class Simulator(object):
    """Simulator base class that manages a full simulation of a node clustering algorithm.

    Attributes:
        nametag (str): the type of the simulated process. Used in the naming of simulation log files.
                        (default value: 'GEN')
        method (int): the clustering method to use for initialization of the network.
                      Used in the iterative algorithms. Possible values {0, 1, 2, 3}
                      representing Spectral, Random, Greedy and NN clustering algorithms respectively.
                      (default value: 1)
        seed (int): seed for the random generator used in the simulations. Forces all algorithms
                    to start with the same random clustering so that we can properly compare them.
                    (default value: 144)
    """

    nametag = "GEN"
    method = 1
    seed = 144

    @classmethod
    def reset_seed(cls):
        """Resets the random seed to value 144.

        Returns:
           no value
        """
        cls.seed = 144

    @classmethod
    def prepare(cls, fstar="new_res_"):
        """Creates the logs path if it does not exist.

        Args:
           fstar (str): the name of the log files directory

        Returns:
           str: the path to the log files
        """
        dir = path.join(LOGS_PATH, fstar)

        if not path.exists(dir):
            mkdir(dir)
        return dir

    @classmethod
    def get_log_file(cls, log_dir, ttype, N, M, d, parts=None):
        """Finds and returns the path of the file to log the simulation results to.

        Args:
            log_dir (str): the directory to store the log file
            ttype (int): traffic type. Available values - see constants - {0,3,5,6,7,8}
            N (int): the number of network nodes
            M (int): the number of clusters
            d (int): the number of nodes in a cluster
            parts (int): the number of cluster membership assignment switches (default value: None)


        Returns:
           str: the path to the log file
        """
        return path.join(cls.prepare(log_dir),
                         str(ttype) + "_" + cls.nametag + "_" + str(N) + "_" + str(M) + "_" + str(d) +
                         ('' if parts == None else '_' + str(parts)) + ".log")

    @classmethod
    def get_data_file(cls, ttype, N, M, d):
        """Finds and returns the path of the random traffic matrix instances .npz file.

        Args:
            ttype (int): traffic type. Available values - see constants - {0,3,5,6,7,8}
            N (int): the number of network nodes
            M (int): the number of clusters
            d (int): the number of nodes in a cluster

        Returns:
           str: the path to the data file
        """

        # flags
        small = N < 64
        big_simulation = ttype in [5, 6] and N > 64

        shuffled = True  # TODO: remove
        data = SH_PREF if shuffled else PREF
        small_data = SH_PREF_SMALL if shuffled else PREF_SMALL

        return path.join(small_data if small else data,
                         DATAFOLDER[ttype],
                         str(N) + "_" + str(M // 2 if big_simulation else M) + "_" + str(
                             2 * d if big_simulation else d) + ".npz")

    @classmethod
    def initialize_clustering(cls, T, M, parts=1):
        """Initializes (randomly by default) the network node's clustering.

        Args:
           T (str): the traffic matrix
           M (int): the number of clusters
           parts (int): the number of cluster membership assignment switches (default value: 1)

        Returns:
           Clusters: the initial clustering of the network
        """
        if cls.method == 1:
            # for every traffic matrix instance on a simulation I use a different seed.
            cls.seed += 1
            random.seed(cls.seed)

        return CLUSTERING_METHOD[cls.method](T, M, parts)

    @classmethod
    def initialize_network(cls, clusters, T):
        """Initializes the network metrics used by the algorithms based on the initial clustering.

        Args:
            clusters (Clusters): the initial clustering of the network nodes
            T (numpy.ndarray): the traffic matrix

        Returns:
            NodeFlows
        """
        return NodeFlows.calculate_flows(clusters, T)

    @classmethod
    def preprocess_traffic(cls, T, transformations=()):
        """Transforms the traffic matrix T  based on the transformations requested.

        Args:
           T (list): the NxN traffic matrix
           transformations (list): a list of transformation functions to apply

        Returns:
           numpy.ndarray: the transformed traffic matrix
        """

        t = np.array(T)
        for transformation, vars in transformations:
            t = transformation(t, *vars)

        return t

    @classmethod
    def improve_clustering(cls, clusters, nf, parts=1):
        """Iteratively improves the node clustering.

        This method is only used for iterative algorithms. If applied on other algorithms
        it has no impact on the initial clustering.

        Args:
           clusters (Clusters): the initial network node's clustering.
           nf (NodeFlows): the initial network flow metrics
           parts (int): the number of cluster membership assignment switches (default value: 1)

        Returns:
           tuple: A tuple of the form (Clusters, NodeFlows, int) representing the clustering and node flows
                  at the end of the process and the number of iterations the iterative process took
        """
        return clusters, nf, 0

    @classmethod
    def run(cls, N, M, d, traffic_type, transformations=(), log_dir='temp', parts=1):
        """Runs a simulation.

         Args:
            N (int): the number of network nodes
            M (int): the number of clusters
            d (int): the number of nodes in a cluster
            traffic_type: traffic type. Available values - see constants - {0,3,5,6,7,8}
            transformations (list): the list of transformations to perform on the traffic matrix
            log_dir (str): the directory to store the log file (default value: 'temp')
            parts (int): the number of cluster membership assignment switches (default value: 1)

         Returns:
            no value
         """

        # file to log the results
        log_file = cls.get_log_file(log_dir, traffic_type, N, M, d, parts)

        # data file with random traffic matrices
        file = cls.get_data_file(traffic_type, N, M, d)

        data = np.load(file)

        logger = Logger(log_file)
        timer = TimingManager(log_file)
        logger.log("--------------------------> Starting simulation")
        logger.log(
            "Nodes: " + str(N) + " Clusters: " + str(M) + " Cluster size: " + str(d) + " Method: " + MESSAGE[
                cls.method] +
            " Parts: " + str(parts), True)

        cls.reset_seed()
        for T in data.values():
            tot, trueintra, trueinter = get_stats(T, traffic_type, N, M, d)

            t = cls.preprocess_traffic(T, transformations)

            logger.log("Clustering")
            with timer:
                clusters = cls.initialize_clustering(t, M, parts)

            nf = cls.initialize_network(clusters, T)

            logger.log("Before >>>")
            logger.log("intra: " + str(nf.intra) + " total: " + str(tot) + " inter: " + str(tot - nf.intra))
            logger.log("Iterative Improvement")
            with timer:
                cl, nf, iter = cls.improve_clustering(clusters, nf, parts)
            logger.log("After >>>")
            logger.log("intra: " + str(nf.intra) + " total: " + str(tot) + " inter: " + str(tot - nf.intra))
            logger.log("Number of iterations: " + str(iter))
            if trueinter == 0:
                trueinter = 'Undefined'
            logger.log("True inter: " + str(trueinter) + "\n")
            logger.log("Clustering: " + str(cl.lookup) + '\n\n')

        logger.log("--------------------------> Ending simulation")


class ISimulator(Simulator):
    """Iterative algorithms simulator base class that manages a full simulation of a node clustering algorithm.
    """

    @classmethod
    def change_method(cls, method):
        """Changes the method to be used for the clustering initialization.

        Args:
            method (int): method used for the clustering initialization.

        Returns:
           no value
        """
        cls.method = method

    @classmethod
    def get_log_file(cls, log_dir, ttype, N, M, d, parts=None):
        """Finds and returns the path of the file to log the simulation results to.

        Args:
            log_dir (str): the directory to store the log file
            ttype (int): traffic type. Available values - see constants - {0,3,5,6,7,8}
            N (int): the number of network nodes
            M (int): the number of clusters
            d (int): the number of nodes in a cluster
            parts (int): the number of cluster membership assignment switches (default value: None)


        Returns:
           str: the path to the log file
        """
        return path.join(cls.prepare(log_dir),
                         str(ttype) + "_" + cls.nametag + MESSAGE[cls.method] + "_" + str(N) + "_" + str(M) + "_" + str(
                             d) +
                         ('' if parts == None else '_' + str(parts)) + ".log")


class BestCandidatesSimulator(ISimulator):
    """Best candidate algorithm simulator that manages a full simulation of the best candidates clustering algorithm.

    Attributes:
        nametag (str): the type of the simulated process. Used in the naming of simulation log files.
                        (default value: 'BC')
        method (int): the clustering method to use for initialization of the network.
                      Used in the iterative algorithms. Possible values {0, 1, 2, 3}
                      representing Spectral, Random, Greedy and NN clustering algorithms respectively.
                      (default value: 1)
    """

    method = 1
    nametag = "BC"

    @classmethod
    def improve_clustering(cls, clusters, nf, parts=1):
        """Iteratively improves the initial given clustering using the stochastic best candidates algorithm.

        Args:
            clusters (Clusters): the initial network node's clustering.
            nf (NodeFlows): the initial network flow metrics
            parts (int): the number of cluster membership assignment switches (default value: 1)

        Returns:
           tuple: A tuple of the form (Clusters, NodeFlows, int) representing the clustering and node flows
                  at the end of the process and the number of iterations the iterative process took
        """
        return stochastic_best_candidates(clusters, nf, parts)

    @classmethod
    def initialize_network(cls, clusters, T):
        """Initializes the network flow metrics used by the algorithms based on the initial clustering.

        Args:
            clusters (Clusters): the initial clustering of the network nodes
            T (numpy.ndarray): the traffic matrix

        Returns:
            NodeFlows
        """
        return NodeFlows.calculate_flows(clusters, T, True)


class RandomSwapsSimulator(ISimulator):
    """Random Swaps algorithm simulator that manages a full simulation of the random swaps clustering algorithm.

    Attributes:
        nametag (str): the type of the simulated process. Used in the naming of simulation log files.
                        (default value: 'RS')
        method (int): the clustering method to use for initialization of the network.
                      Used in the iterative algorithms. Possible values {0, 1, 2, 3}
                      representing Spectral, Random, Greedy and NN clustering algorithms respectively.
                      (default value: 1)
    """

    method = 1
    nametag = "RS"

    @classmethod
    def improve_clustering(cls, clusters, nf, parts=1):
        """Iteratively improves the initial given clustering using the iterative random swaps algorithm.

        Args:
            clusters (Clusters): the initial network node's clustering.
            nf (NodeFlows): the initial network flow metrics
            parts (int): the number of cluster membership assignment switches (default value: 1)

        Returns:
           tuple: A tuple of the form (Clusters, NodeFlows, int) representing the clustering and node flows
                  at the end of the process and the number of iterations the iterative process took
        """
        return iterative_random_swaps(clusters, nf, parts)


class GreedySimulator(Simulator):
    """Greedy clustering algorithm simulator that manages a full simulation of greedy clustering algorithm.

    Attributes:
        nametag (str): the type of the simulated process. Used in the naming of simulation log files.
                        (default value: 'G')
        method (int): the clustering method to use for initialization of the network.
                      Used in the iterative algorithms. Possible values {0, 1, 2, 3}
                      representing Spectral, Random, Greedy and NN clustering algorithms respectively.
                      (default value: 2)
    """
    method = 2
    nametag = "G"


class NNSimulator(Simulator):
    """Nearest Neighbors clustering algorithm simulator that manages a full simulation of NN clustering algorithm.

    Attributes:
        nametag (str): the type of the simulated process. Used in the naming of simulation log files.
                        (default value: 'NN')
        method (int): the clustering method to use for initialization of the network.
                      Used in the iterative algorithms. Possible values {0, 1, 2, 3}
                      representing Spectral, Random, Greedy and NN clustering algorithms respectively.
                      (default value: 3)
    """
    method = 3
    nametag = "NN"


class SpectralSimulator(Simulator):
    """Spectral clustering algorithm simulator that manages a full simulation of spectral clustering algorithm.

    Attributes:
        nametag (str): the type of the simulated process. Used in the naming of simulation log files.
                        (default value: 'S')
        method (int): the clustering method to use for initialization of the network.
                      Used in the iterative algorithms. Possible values {0, 1, 2, 3}
                      representing Spectral, Random, Greedy and NN clustering algorithms respectively.
                      (default value: 0)
    """
    method = 0
    nametag = "S"


def simulate(fname="temp", simulators=((GreedySimulator, ()),), ttypes=(0, ), running_mode=(False, False, True)):
    """Conducts a simulation. The simulation results get written in the log files directory given.

    Args:
        fname (str): the name of the logs directory for the simulation (default value: 'temp')
        simulators (list): a list of the algorithms to simulate along with the appropriate transformations to apply
                           (default value: GreedySimulator with no transformations)
        ttypes (list): a list of the traffic types to run the simulations on.
                       (default value: unit-out-clustered code 0 )
        running_mode (tuple): a list of flags of the form <small, big, parts> that indicates whether we want
                             to run the simulation on small instances, big instances and whether we want to
                             simulate multiple switches in the cluster membership assignment.
                             (default value: (False, False, True) )
    Returns:
       no value
    """
    # flags
    run_small, run_big, run_parts = running_mode

    # small instances

    if run_small:
        small_sizes = [8, 16, 32]

        for N in small_sizes:
            for M in [2, 4, 8]:
                d = N // M
                if d > 2:
                    for ttype in ttypes:
                        for sim, transf in simulators:
                            sim.run(N, M, d, ttype, transf, fname)

    # big instances

    if run_big:
        big_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

        d = 32
        for N in big_sizes:
            M = ceil(N / d)
            for ttype in ttypes:
                if ttype in [5, 6]:
                    if N == 64: continue
                for sim, transf in simulators:
                    sim.run(N, M, d, ttype, transf, fname)

    # partitions

    if run_parts:
        d = 32
        N = 1024
        M = ceil(N / d)

        # at the worst case I have as many parts as the number of nodes in a cluster cause each one
        # provides one node for each cluster
        parts = 1
        while parts < 2:
            for ttype in ttypes:
                for sim, transf in simulators:
                    sim.run(N, M, d, ttype, transf, fname, parts)
            parts *= 2


def main():
    """Main function to run the simulation.

    This function is expected to change in every simulation.
    A better future alternative is command line simulation inputs.

    Returns:
        no value
    """
    # TODO (gina) : command line  simulation inputs

    fname = "simulation_greedy_8_may"

    mode = (False, True, False)
    simulators = [(GreedySimulator, []), ]
    #
    # (GreedySimulator, []),
    # (RandomSwapsSimulator, []),
    # (BestCandidatesSimulator, []),
    traffics = [3, 0, 6, 8]
    simulate(fname, simulators, traffics, mode)

    summarize_results(fname)


if __name__ == "__main__":

    main()

