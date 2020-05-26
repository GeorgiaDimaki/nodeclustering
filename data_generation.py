"""Functions that generate traffic matrices used for the simulations.

Files generated are compressed numpy array files (.npz) that contain random
traffic matrix instances each. Each traffic type has a different code and
dedicated function for the random instances generation. All instances have
randomly shuffled rows(/columns) so that block patterns are not obvious to
detect by the algorithms.

Traffic type codes:

    0: unit out clustered
    3: non clustered
    5: random clustered controlled
    6: controlled dense
    7: all to all unit traffic
    8: dense clustered


File naming convention:

    data_shuffled/<traffic type folder name>/N_M_d.npz

"""

from math import *
from constants import *


def empty_self_traffic(t):
    """Sets self traffic to zero.

    Args:
        t (numpy.ndarray): the traffic matrix to change

    Returns:
        t (numpy.ndarray)
    """
    for i in range(len(t)):
        t[i, i] = 0
    return t


def unit_out_clustered(N, d):
    """Generates a clustered traffic matrix of unit out clustered traffic.

    Intra cluster traffic flows take values uniformly at random in [7,10].
    All inter cluster traffic flows have value of 1 unit.
    code: 0

    Args:
        N (int): number of nodes
        d (int): number of nodes per cluster

    Returns:
        numpy.ndarray
    """
    t = np.ones((N, N))
    k = 0
    l = d
    for i in range(N // d):
        t[k:l, k:l] = np.random.randint(low=7, high=10, size=(d, d))
        k += d
        l += d

    rem = N % d
    if rem != 0:
        k = N - rem
        t[k:N, k:N] = np.random.randint(7, 10, size=(rem, rem))

    t = empty_self_traffic(t)
    return t


def nonclustered(N, d):
    """Generates a traffic matrix of non clustered random traffic.

    All traffic flows take values uniformly at random in [1,10].
    code: 3

    Args:
        N (int): number of nodes
        d (int): number of nodes per cluster

    Returns:
        numpy.ndarray
    """
    t = np.random.randint(1, 10, size=(N, N))
    t = empty_self_traffic(t)
    return t


def random_clustered_controlled(N, d=64):
    """Generates traffic matrix of type random clustered controlled.

    This traffic is clustered in two levels. In clusters of size d/2 -default 32- (small blocks)
    and broader clusters of size d -default value 64- (bigger blocks). Flows in the d/2 sized
    clusters take values uniformly at random in [7,10]. Flows that belong in the d/2 clusters
    but not in the d take value 6. The smaller general flows take values in [1,3] uniformly at
    random. This helps us calculate the true optimal (sum of values less than 7).
    code: 5

    Args:
        N (int): number of nodes
        d (int): number of nodes per cluster

    Returns:
        numpy.ndarray
    """
    t = np.random.randint(1, 3, size=(N, N))
    d = 2 * d
    k = 0
    l = d
    for i in range(N // d):
        t[k:l, k:l] = np.ones((d, d)) * 6
        t[k:(l - (d // 2)), k:(l - (d // 2))] = np.random.randint(7, 10, size=(d // 2, d // 2))
        t[(l - (d // 2)):l, (l - (d // 2)):l] = np.random.randint(7, 10, size=(d // 2, d // 2))
        k += d
        l += d

    rem = N % d
    k = N - rem
    if rem != 0:
        t[k:N, k:N] = np.random.randint(7, 10, size=(rem, rem))

    t = empty_self_traffic(t)
    return t


def random_clustered_controlled_dense(N, d=64):
    """Generates traffic matrix of type random clustered controlled.

    This traffic is clustered in two levels. In clusters of size d/2 -default 32- (small blocks)
    and broader clusters of size d -default value 64- (bigger blocks). Flows in the d/2 sized
    clusters take values uniformly at random in [7,8]. Flows that belong in the d/2 clusters
    but not in the d take values uniformly at random in (6,7]. The smaller general flows take
    values in [1,6] uniformly at random. This helps us calculate the true optimal (sum of
    values less than 7).
    code: 6

    Args:
        N (int): number of nodes
        d (int): number of nodes per cluster

    Returns:
        numpy.ndarray
    """
    t = (6 - 1) * np.random.random_sample((N, N)) + 1
    d = 2 * d
    k = 0
    l = d
    for i in range(N // d):
        t[k:l, k:l] = np.ones((d, d)) * 7 - np.random.random_sample((d, d))
        t[k:(l - (d // 2)), k:(l - (d // 2))] = np.random.random_sample(size=(d // 2, d // 2)) + 7
        t[(l - (d // 2)):l, (l - (d // 2)):l] = np.random.random_sample(size=(d // 2, d // 2)) + 7
        k += d
        l += d

    rem = N % d
    k = N - rem
    if rem != 0:
        t[k:N, k:N] = np.random.random_sample(size=(rem, rem)) + 7

    t = empty_self_traffic(t)
    return np.round(t, 2)


def alltoall(N, d):
    """Generates uniform all to all traffic flows matrix.

    All traffic flows are set to 1 unit.
    code: 7

    Args:
        N (int): number of nodes
        d (int): number of nodes per cluster

    Returns:
        numpy.ndarray
    """
    t = np.ones((N, N), dtype=int)
    t = empty_self_traffic(t)
    return t


def dense_clustered(N, d):
    """Generates a dense clustered traffic matrix.

    Intra cluster traffic flows take values uniformly at random in [7,8].
    All inter cluster traffic flows have values uniformly at random in [6,7).
    code: 8

    Args:
        N (int): number of nodes
        d (int): number of nodes per cluster

    Returns:
        numpy.ndarray
    """
    t = 7 - 1 * np.random.random_sample((N, N))
    k = 0
    l = d
    for i in range(N // d):
        t[k:l, k:l] = np.random.randint(low=7, high=8, size=(d, d))
        k += d
        l += d

    rem = N % d
    if rem != 0:
        k = N - rem
        t[k:N, k:N] = np.random.randint(7, 8, size=(rem, rem))

    t = empty_self_traffic(t)
    return np.round(t, 2)


def generate(type, N, d, instances, foldername='data'):
    """Generates random traffic matrix data.

    Generates <instances> random instances of traffic matrix of type <type>
    for a network of size N and cluster's size d and saves them in a folder of
    name <foldername>.

    Args:
        type (int): the traffic type to generate {0,3,5,6,7,8}
        N (int): number of nodes
        d (int): number of nodes per cluster
        instances (int): number of traffic matrix instances to generate
        foldername (str): the name of the folder to save the data generated (default value: 'data')

    Returns:
        no value
    """
    funcs = {0: unit_out_clustered, 3: nonclustered, 5: random_clustered_controlled,
             6: random_clustered_controlled_dense, 7: alltoall, 8: dense_clustered}

    tfolder = {0: "unit_out_clustered", 3: "nonclustered", 5: "controlled",
              6: "controlled_dense", 7: "alltoall", 8: "dense_clustered"}

    inst = {}
    for i in range(instances):
        array = funcs[type](N, d)
        indices = np.array(range(len(array)))
        np.random.shuffle(indices)
        array = array[indices]
        array = array.transpose()[indices]
        array = array.transpose()
        inst[str(i)] = array

    M = ceil(N / d)
    name = str(N) + "_" + str(M) + "_" + str(d)

    folder = os.path.join(foldername, tfolder[type])
    if not os.path.exists(folder):
        os.mkdir(folder)
    filename = os.path.join(folder, name)

    np.savez_compressed(filename, **inst)


if __name__ == '__main__':

    instances = 25

    # --- big instances

    d = 32
    code = 5
    for N in [64, 128, 256, 512, 1024, 2048, 4096]:
        if code in [5, 6]:
            d = 64
        generate(code, N, d, instances, 'data_shuffled')

    # --- small instances

    folder = os.path.join('data_shuffled', 'small_instances')
    for N in [8, 16, 32]:
        M = 2
        d = N // M
        while d > 2:
            generate(code, N, d, instances, folder)
            M *= 2
            d = N // M
