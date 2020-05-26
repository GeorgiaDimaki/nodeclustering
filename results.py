"""Functions to extract the results from the log files.

The same logging conventions are used by the simulation functions.
The result extraction functions are dedicated to read the format of the logs
and extract the needed information.

Standard format of log files:

    --------------------------> Starting simulation
    [TIMESTAMP] 2020-02-26 09:22:09.569194
    Nodes: 32 Clusters: 8 Cluster size: 4 Method: G

    Clustering
    [E] 0:00:00.000296
    Before >>>
    intra: 482.0 total: 1668.0 inter: 1186.0
    Iterative Improvement
    [E] 0:00:00.012947
    After >>>
    intra: 649.0 total: 1668.0 inter: 1019.0
    Number of iterations: 10
    True inter: 896.0


    Clustering: [0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 5, 5, 5, 5, 0, 0, 0, 6, 6, 6, 6, 7, 7, 7, 7]


    --------------------------> Ending simulation

or more abstractly:

    --------------------------> Starting simulation
    <General simulation information>

    <instance 1 simulation>
    <instance 2 simulation>
            ...
    --------------------------> Ending simulation
"""

import glob
import os
from datetime import date
import re
from math import sqrt
from json import dump

# CONSTANTS
from constants import DATAFOLDER, CLUSTER_SIZE, RES_PATH

NUMBER = re.compile("[-+]?\d*\.\d+|\d+")


def parse_time(times):
    """Parses strings that represent time and translates them into number of seconds.

    Args:
        times (list): a list of strings to translate into number of seconds

    Returns:
        list: the list of parse times
    """
    new_times = []
    for t in times:
        f = t.split(":")
        new_times.append(int(f[0]) * 3600 + int(f[1]) * 60 + float(f[2]))
    return new_times


def parse_float(array):
    """Casts all values in the array into floats.

    Args:
        array (list): a list to convert into list of floats

    Returns:
        list: the converted list
    """
    return [float(a) for a in array]


def parse_int(array):
    """Casts all values in the array into ints.

    Args:
        array (list): a list to convert into list of ints

    Returns:
        list: the converted list
    """
    return [int(a) for a in array]


def get_avg_time(timespans):
    """Calculates the average of the numbers in the timespans list.

    Args:
        timespans (list): a list of timespans

    Returns:
        float: the average timespan
    """
    return sum(timespans) / len(timespans)


def get_info(filepath):
    """Extracts the simulation information from the respective log filename.

    The name of the file is structured as: ttype_method_N_M_d_(optional: parts).npz

    Args:
        filepath (str): the file to extract information from

    Returns:
        tuple: a tuple of the form (ttype, method, N, M, d, parts)
    """
    filename = filepath[filepath.rfind(os.path.sep) + 1:]
    without_ext = filename.replace('.log', '')
    info = without_ext.split('_')
    return int(info[0]), info[1], int(info[2]), int(info[3]), int(info[4]), \
           None if len(info) == 5 else int(info[5])


def read_instance_simulation(f):
    """Reads the simulation information of a traffic matrix instance.

    Example of the format read:

    Clustering
    [E] 0:00:00.000296
    Before >>>
    intra: 482.0 total: 1668.0 inter: 1186.0
    Iterative Improvement
    [E] 0:00:00.012947
    After >>>
    intra: 649.0 total: 1668.0 inter: 1019.0
    Number of iterations: 10
    True inter: 896.0


    Clustering: [0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 5, 5, 5, 5, 0, 0, 0, 6, 6, 6, 6, 7, 7, 7, 7]

    Args:
        f (file): the file to extract information from

    Returns:
        tuple: a tuple of the form
                ([time_c, time_i, n_iter, inter_s, inter_e, intra_s, intra_e, total, true_inter, clustering], f)
    """
    line = f.readline()
    while line == '\n':
        line = f.readline()

    # clustering
    if 'Clustering' not in line: return None, f
    time_c = f.readline().strip('\n').split(' ')[1]

    if 'Before' not in f.readline():
        return None, f
    intra_s, total, inter_s = NUMBER.findall(f.readline().strip('\n'))
    if f.readline().strip('\n') not in ['Iterative Improvement', 'Stochastic Improvement', 'Bayesian Improvement']:
        return None, f

    # iterative improvement
    time_i = f.readline().strip('\n').split(' ')[1]
    if 'After' not in f.readline().strip('\n'):
        return None, f
    intra_e, _, inter_e = NUMBER.findall(f.readline().strip('\n'))

    # number of iterations
    n_iter = NUMBER.findall(f.readline().strip('\n'))[0]
    line = f.readline()

    # true inter cluster traffic
    true_inter = None if 'Undefined' in line else NUMBER.findall(line.strip('\n'))[0]
    line = f.readline()
    while line == '\n':
        line = f.readline()
    line = line.replace("Clustering: ", "")
    while ']' not in line:
        line = line + f.readline().strip('\n')
    line = line.strip(']\n').strip('[')
    clustering = line.split(',') if ',' in line else line.split(' ')

    return [time_c, time_i, n_iter, inter_s, inter_e, intra_s, intra_e, total, true_inter, clustering], f


def read_datetime(line):
    """Reads the timestamp of the simulation start available in the general information of the log file.

    Example of the format read:

    [TIMESTAMP] 2020-02-26 09:22:09.569194

    Args:
        line (str): the line that contains the timestamp

    Returns:
        str: the timestamp
    """
    if "[TIMESTAMP]" not in line:
        return None
    line = line.replace('[TIMESTAMP]', '')
    return line.strip()


def read_general_info(line):
    """Reads the general information of the simulation.

    Example of the format read:

    Nodes: 32 Clusters: 8 Cluster size: 4 Method: G Parts: 1

    0       1   2       3   4       5   6   7     8 9      10  >> indices in words variable

    Args:
        line (str): the line that contains the information

    Returns:
        tuple: a tuple of the form (N, M, d, method, parts)
    """
    words = line.split(" ")
    parts = None if len(words) < 11 else int(words[10])
    return int(words[1]), int(words[3]), int(words[6]), words[8], parts


def read_simulation(f, ttype, method, write=True):
    """Reads a full simulation that starts with 'Starting" and ends with 'Ending'.

    Since we might have simulated more than 1 traffic matrix instances, each instance
    is parsed and the respective information are saved in lists. If write is True then
    the information of each instance is also written as a row in a csv file. At the end
    all lists are returned for further calcultions.

    Args:
        f (file): the log file of the simulation
        ttype (int): the traffic type
        method (int): the method used for the initial clustering
        write (bool): whether or not to write the data in a csv

    Returns:
        tuple: a tuple of the form
              (clustering, iimprovement, iter, inter_start, inter_end, intra_start,
              intra_end, total, true_inter, true_intra, f)
    """

    first_line = f.readline()
    while "Starting" not in first_line:
        first_line = f.readline()
    datetime = read_datetime(f.readline())
    N, M, d, _, parts = read_general_info(f.readline().strip('\n'))

    iimprovement = []
    clustering = []
    inter_start = []
    inter_end = []
    intra_start = []
    intra_end = []
    true_inter = []
    true_intra = []
    total = []
    iter = []

    instance = 0
    binder = '$'
    newline = '\n'
    to_read = f.tell()

    while 'Ending' not in f.readline():
        f.seek(to_read)
        info, f = read_instance_simulation(f)

        clustering.append(info[0])
        time = (info[0]).split(":")
        info[0] = int(time[0]) * 3600 + int(time[1]) * 60 + float(time[2])

        iimprovement.append(info[1])
        time = (info[1]).split(":")
        info[1] = int(time[0]) * 3600 + int(time[1]) * 60 + float(time[2])

        iter.append(info[2])
        inter_start.append(info[3])
        inter_end.append(info[4])
        intra_start.append(info[5])
        intra_end.append(info[6])
        total.append(info[7])
        true_inter.append(info[8])

        if info[8] is None:
            true_intra.append(None)
        else:
            true_intra.append(str(float(info[7]) - float(info[8])))

        to_read = f.tell()
        while f.readline() == newline:
            to_read = f.tell()

        f.seek(to_read)

        simulations_csv = 'simulations.csv'

        # headline:
        # Datetime, N, M, d, parts, ttype, method, instance, time_c, time_i, n_iter,
        # inter_s, inter_e, intra_s, intra_e, total, true_inter, clustering, parts

        if write:
            with open(simulations_csv, 'a') as file:
                to_write = [datetime, str(N), str(M), str(d), str(parts),
                            str(ttype), method, str(instance), binder.join([str(i) for i in info])]
                file.write(binder.join(to_write) + "\n")

        instance += 1

    return clustering, iimprovement, iter, inter_start, inter_end, intra_start, intra_end, total, \
           true_inter, true_intra, f


def get_results(filename):
    """Calculates the results of interest of a simulation log file.

    It extracts all the results from a file and then performs post processing
    to calculate averages, min and max of the simulation results.
    Given that we have many random traffic matrix instances, to evaluate
    the algorithms performance we aggregate the results of the simulations on
    the multiple instances.

    Args:
        filename (str): the name of the log file

    Returns:
        tuple: a tuple that contains
               (avg clustering time, max clustering time, min clustering time, sd of clustering time,
               avg iterative improvement time (ii), max ii, min ii, sd ii,
               avg total traffic (tot), max tot, min tot, sd tot,
               avg iter cluster traffic (inter), min iter, max iter, sd iter,
               avg optimality gap (optgap), min optgap, max optgap, sd optgap,
               avg optimal value percentage (optper), min optper, max optper, sd optper,
               avg improvement percentage (impper), min impper, max impper, sd impper)

    """

    ttype, method, N, M, d, parts = get_info(filename)

    with open(filename) as f:
        time_c, time_i, iter, inter_s, inter_e, intra_s, intra_e, tot, tinter, tintra, f = read_simulation(f, ttype,
                                                                                                           method)
        time_c = parse_time(time_c)
        time_i = parse_time(time_i)
        iter = parse_int(iter)
        inter_s = parse_float(inter_s)
        inter_e = parse_float(inter_e)
        intra_s = parse_float(intra_s)
        intra_e = parse_float(intra_e)
        tot = parse_float(tot)
        tinter = tinter if None in tinter else parse_float(tinter)
        tintra = tintra if None in tintra else parse_float(tintra)

        instances = len(time_c)

        avg_cl = get_avg_time(time_c)
        min_cl = min(time_c)
        max_cl = max(time_c)
        sd_cl = sqrt(sum((t - float(avg_cl)) ** 2 for t in time_c) / instances)

        avg_ii = get_avg_time(time_i)
        min_ii = min(time_i)
        max_ii = max(time_i)
        sd_ii = sqrt(sum((t - float(avg_ii)) ** 2 for t in time_i) / instances)

        time_tot = time_c + time_i
        avg_tot = get_avg_time(time_tot)
        min_tot = min(time_tot)
        max_tot = max(time_tot)
        sd_tot = sqrt(sum((t - float(avg_tot)) ** 2 for t in time_tot) / instances)

        avg_iter = sum(iter) / instances
        min_iter = min(iter)
        max_iter = max(iter)
        sd_iter = sqrt(sum((t - float(avg_iter)) ** 2 for t in iter) / instances)

        if None in tinter:
            avg_optgap = min_optgap = max_optgap = sd_optgap = None
        else:
            opt_gap = []
            for i in range(len(intra_e)):
                opt_gap.append(abs(float(tintra[i]) - intra_e[i]) / intra_e[i])

            avg_optgap = sum(opt_gap) / instances
            min_optgap = min(opt_gap)
            max_optgap = max(opt_gap)
            sd_optgap = sqrt(sum((t - float(avg_optgap)) ** 2 for t in opt_gap) / instances)

        if None in tinter:
            avg_optper = min_optper = max_optper = sd_optper = None
        else:
            opt_per = []
            for i in range(len(intra_e)):
                opt_per.append(abs(float(tintra[i]) - intra_e[i]) / float(tintra[i]))

            avg_optper = sum(opt_per) / instances
            min_optper = min(opt_per)
            max_optper = max(opt_per)
            sd_optper = sqrt(sum((t - float(avg_optper)) ** 2 for t in opt_per) / instances)

        improvement_per = []
        for i in range(len(intra_e)):
            improvement_per.append(abs(intra_s[i] - intra_e[i]) / intra_s[i])

        avg_impper = sum(improvement_per) / instances
        min_impper = min(improvement_per)
        max_impper = max(improvement_per)
        sd_impper = sqrt(sum((t - float(avg_impper)) ** 2 for t in improvement_per) / instances)

        return avg_cl, max_cl, min_cl, sd_cl, \
               avg_ii, max_ii, min_ii, sd_ii, avg_tot, max_tot, min_tot, sd_tot, avg_iter, min_iter, \
               max_iter, sd_iter, avg_optgap, min_optgap, max_optgap, sd_optgap, avg_optper, min_optper, \
               max_optper, sd_optper, avg_impper, min_impper, max_impper, sd_impper


def summarize_results(res_dir):
    """Summarizes the results of a simulation.

    It extracts the results of all the log files in a simulation's log directory.
    It saves the results both in a json format file and in a csv file.

    Args:
        res_dir (str): the simulation logs directory

    Returns:
        no value
    """

    print("Gathering Results...")

    with open("simulations.csv", 'a') as f:
        f.write('Datetime $ N $ M $ d $ parts $ ttype $ method $ instance $ time_c $ time_i $ n_iter $ '
                'inter_s $ inter_e $ intra_s $ intra_e $ total $ true_inter $ clustering\n')

    results = {}
    for k in CLUSTER_SIZE:
        results[k] = {}
        for l in DATAFOLDER.values():
            results[k][l] = {}
            for m in ['G', 'R', 'S', 'BCS', 'BCG', 'BCR', 'RSR', 'RSG', 'RSS', 'NN']:
                results[k][l][m] = {'N': [], 'avg_cl': [], 'max_cl': [], 'min_cl': [], 'sd_cl': [],
                                    'avg_ii': [], 'max_ii': [], 'min_ii': [], 'sd_ii': [], 'avg_tot': [],
                                    'max_tot': [], 'min_tot': [], 'sd_tot': [], 'avg_iter': [], 'min_iter': [],
                                    'max_iter': [], 'sd_iter': [], 'avg_optgap': [], 'min_optgap': [], 'max_optgap': [],
                                    'sd_optgap': [], 'avg_optper': [], 'min_optper': [], 'max_optper': [],
                                    'sd_optper': [],
                                    'avg_impper': [], 'min_impper': [], 'max_impper': [], 'sd_impper': []}

    log_dir = os.path.join(os.getcwd(), "..", "Logs", res_dir, "*.log")
    logs = glob.glob(log_dir)

    for f in logs:
        ttype, method, N, M, d, parts = get_info(f)
        print(get_info(f))
        info = get_results(f)

        res = results[d][DATAFOLDER[int(ttype)]][method]
        res['N'].append(N if int(N) >= 64 else str(N) + '_' + str(M))
        res['avg_cl'].append(info[0])
        res['max_cl'].append(info[1])
        res['min_cl'].append(info[2])
        res['sd_cl'].append(info[3])
        res['avg_ii'].append(info[4])
        res['max_ii'].append(info[5])
        res['min_ii'].append(info[6])
        res['sd_ii'].append(info[7])
        res['avg_tot'].append(info[8])
        res['max_tot'].append(info[9])
        res['min_tot'].append(info[10])
        res['sd_tot'].append(info[11])
        res['avg_iter'].append(info[12])
        res['min_iter'].append(info[13])
        res['max_iter'].append(info[14])
        res['sd_iter'].append(info[15])
        res['avg_optgap'].append(info[16])
        res['min_optgap'].append(info[17])
        res['max_optgap'].append(info[18])
        res['sd_optgap'].append(info[19])
        res['avg_optper'].append(info[20])
        res['min_optper'].append(info[21])
        res['max_optper'].append(info[22])
        res['sd_optper'].append(info[23])
        res['avg_impper'].append(info[24])
        res['min_impper'].append(info[25])
        res['max_impper'].append(info[26])
        res['sd_impper'].append(info[27])

    res_filename = os.path.join(os.getcwd(), RES_PATH, res_dir + "__" + str(date.today()))
    fjson = res_filename + ".json"
    fcsv = res_filename + ".csv"
    print("Saving Results in : " + res_filename)

    with open(fjson, 'w') as fp:
        dump(results, fp, indent=4)

    if os.path.exists("simulations.csv"):
        os.rename("simulations.csv", fcsv)

