"""Constants.

Some of the constants support the functionality of the algorithms and define traffic types or
central directories for logging or result publishing.

Some other constants act as definitions of the problem's domain.
They are unused but might be used in the future.
The information on the domain have been found in the following sources:

> AWGR Switches sizes:

https://www.osapublishing.org/DirectPDFAccess/A922A930-EB74-6656-875B5C4EF617489C_382251/oe-26-5-6276.pdf?da=1&id=382251&seq=0&mobile=no
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6691912
https://www.researchgate.net/publication/3385456_64_64-channel_uniform-loss_and_cyclic-frequency_arrayed-waveguide_grating_router_module
https://www.researchgate.net/publication/258655460_Large_Port_Count_Wavelength_Routing_Optical_Switch_Consisting_of_Cascaded_Small-Size_Cyclic_Arrayed_Waveguide_Gratings
https://ieeexplore.ieee.org/document/5199357

> Optical Switch sizes:

https://ciscodatacenter.files.wordpress.com/2009/07/data-center-top-of-rack-architecture-design.pdf
https://www.cisco.com/c/en/us/products/switches/nexus-3000-series-switches/models-comparison.html#~tab-nexus3600

> Rack sizes:

# max rack size 256 ---> I will assume that the typical rack size is 48-128
# 8, 16, 32, 40, 42
https://www.webhostingtalk.com/showthread.php?t=673785

https://blogs.cisco.com/datacenter/5-features-to-consider-when-buying-a-top-of-rack-switch
https://docs.vmware.com/en/VMware-Cloud-Foundation/2.2/com.vmware.vcf.ovdeploy.doc_22/GUID-A5058F56-B383-49B1-B57E-B2C7875D1196.html

"""


import os
from clustering import *

LOGS_PATH = os.path.join("..", "Logs")
RES_PATH = os.path.join(".", "res")

SIMULATIONS_CSV = os.path.join("..", "Results", "simulations.csv")

PREF = "data"
PREF_SMALL = os.path.join(PREF, 'small_instances')

SH_PREF = "data_shuffled"
SH_PREF_SMALL = os.path.join(SH_PREF, 'small_instances')

DATAFOLDER = {0: "unit_out_clustered", 3: "nonclustered",
              5: "controlled", 6: "controlled_dense",
              7: "alltoall", 8:"dense_clustered"}

CLUSTERING_METHOD = {0: fast_clustering , 1: slow_clustering, 2: greedy_clustering, 3: nn_clustering}
MESSAGE = {-2: 'NONE',
           -1: "ALL",
           0: "S",
           1: "R",
           2: "G",
           3: "NN"}

AWGR_SIZE = 32
CLUSTER_SIZE = [4, 8, 16, 32, 64, 128, 256, 512]

NETWORK_SIZE = [64, 128, 256, 512, 1024, 2048, 4096]

SWITCH_SIZE = 4096

RACK_SIZE = 40

# when the size is less than that then we are performing a small instance test
MINIMUM_NETWORK_SIZE = 64
