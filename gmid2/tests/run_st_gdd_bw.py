import os
import sys
import time
from gmid2.global_constants import *
from gmid2.basics.uai_files import read_limid, read_svo
from gmid2.basics.directed_network import DecisionNetwork
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.submodel import submodel_tree_decomposition
from gmid2.inference.st_gdd_bw import StGDDBw


def run(TEST_PATH, alpha, ibound, prefix):
    print("algorithm\t\t{}".format(StGDDBw.__name__))
    print("ibound\t\t{}".format(ibound))
    print("alpha\t\t{}".format(alpha))
    print("\n\n")

    for f in sorted(os.listdir(TEST_PATH)):
        if f.endswith(".uai") and f.startswith(prefix):
            print("\nSTRAT {}\t\t{}".format(f, time.ctime(time.time())))
            file_name = os.path.join(TEST_PATH, f.replace(".uai", ""))
            file_info = read_limid(file_name, skip_table=False)

            t0 = time.time()
            gm = GraphicalModel()
            gm.build(file_info)
            gm.convert_prob_to_log()    # conversion is done here!
            gm.convert_util_to_alpha(alpha)
            print("gm\t\t{}".format(time.time()-t0))

            t0 = time.time()
            dn = DecisionNetwork()
            dn.build(file_info)
            print("dn\t\t{}".format(time.time() - t0))

            t0 = time.time()
            st = submodel_tree_decomposition(dn)
            print("st\t\t{}".format(time.time() - t0))
            print("roots\t\t{}".format(len(dn.value_nids)+1))

            st_gdd = StGDDBw(gm, st, i_bound=ibound, alpha=alpha)  # __init__ for message passing and nx.DiGraph
            t0 = time.time()
            st_gdd.build_message_graph()
            print("build\t\t{}".format(time.time()-t0))
            st_gdd.schedule()
            st_gdd.init_propagate()
            t0 = time.time()
            st_gdd.propagate_iter()
            bound = st_gdd.bounds()
            print("prop\t\t{}".format(time.time()-t0))
            print("ub\t\t{}".format(bound))
            print("END {}\t\t{}".format(f, time.ctime(time.time())))


if __name__ == "__main__":
    domain = sys.argv[1]
    alpha = float(sys.argv[2])
    ibound = int(sys.argv[3])
    try:
        prefix = sys.argv[4]
    except:
        prefix = ""
    run(os.path.join(BENCHMARK_DIR, domain), alpha, ibound, prefix)