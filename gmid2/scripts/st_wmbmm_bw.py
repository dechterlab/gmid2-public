PRJ_PATH = "/home/junkyul/conda/gmid2"
import sys
sys.path.append(PRJ_PATH)

import os
import time
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.global_constants import *
from gmid2.basics.uai_files import read_limid, read_svo
from gmid2.basics.directed_network import DecisionNetwork
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.submodel import submodel_tree_decomposition
from gmid2.inference.st_wmbmm_bw import StWMBMMBw


def run(file_path, ibound):
    print("{}\t\t{}".format(StWMBMMBw.__name__, ibound))
    f = file_path.split("/")[-1].replace(".uai", "")
    print("\nSTART {}\t\t{}".format(f, time.ctime(time.time())))
    file_name = file_path.replace(".uai", "")
    file_info = read_limid(file_name, skip_table=False)

    gm = GraphicalModel()
    gm.build(file_info)
    gm.convert_prob_to_log()  # conversion is done here!
    gm.convert_util_to_alpha(1.0)

    dn = DecisionNetwork()
    dn.build(file_info)

    t0 = time.time()
    st = submodel_tree_decomposition(dn)
    print("st\t\t{}".format(time.time()-t0))
    print("roots\t\t{}".format(len(dn.value_nids) + 1))

    st_mp = StWMBMMBw(gm, st, i_bound=ibound, alpha=1.0)  # __init__ for message passing and nx.DiGraph
    t0 = time.time()
    st_mp.build_message_graph()
    print("build\t\t{}".format(time.time() - t0))
    st_mp.schedule()
    st_mp.init_propagate()
    t0 = time.time()
    st_mp.propagate_iter()
    bound = st_mp.bounds()
    print("prop\t\t{}".format(time.time() - t0))
    print("ub\t\t{}".format(bound))
    print("END {}\t\t{}".format(f, time.ctime(time.time())))
    return bound


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        ibound = int(sys.argv[2])
    else:
        TEST_PATH = os.path.join(BENCHMARK_DIR, "synthetic")
        f = "mdp1-4_2_2_5.uai"
        file_path = os.path.join(TEST_PATH, f)
        ibound = 1

    run(file_path, ibound)
