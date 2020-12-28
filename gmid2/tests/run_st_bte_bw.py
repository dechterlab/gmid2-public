import os
import sys
import time
from gmid2.global_constants import *
from gmid2.basics.uai_files import read_limid
from gmid2.basics.directed_network import DecisionNetwork
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.submodel import submodel_tree_decomposition
from gmid2.inference.st_bte_bw import StBTEBw

def run(TEST_PATH, alpha, prefix):
    print("algorithm\t\t{}".format(StBTEBw.__name__))
    print("alpha\t\t{}".format(alpha))
    print("\n\n")

    for f in sorted(os.listdir(TEST_PATH)):
        if f.endswith(".uai") and f.startswith(prefix):
            print("STRAT {}".format(f))
            file_name = os.path.join(TEST_PATH, f.replace(".uai", ""))
            file_info = read_limid(file_name, skip_table=False)

            t0 = time.time()
            gm = GraphicalModel()
            gm.build(file_info)
            gm.convert_prob_to_log()  # conversion is done here!
            gm.convert_util_to_alpha(alpha)
            print("gm\t\t{}".format(time.time() - t0))

            t0 = time.time()
            dn = DecisionNetwork()
            dn.build(file_info)
            print("dn\t\t{}".format(time.time() - t0))

            t0 = time.time()
            st = submodel_tree_decomposition(dn)
            print("st\t\t{}".format(time.time() - t0))
            print("roots\t\t{}".format(len(dn.value_nids) + 1))

            st_mp = StBTEBw(gm, st, alpha=alpha)  # __init__ for message passing and nx.DiGraph
            t0 = time.time()
            st_mp.build_message_graph()
            print("build\t\t{}".format(time.time()-t0))
            st_mp.schedule()
            st_mp.init_propagate()
            t0 = time.time()
            st_mp.propagate_iter()
            bound = st_mp.bounds()
            print("prop\t\t{}".format(time.time()-t0))
            print("ub\t\t{}".format(bound))
            print("END {}".format(f))


if __name__ == "__main__":
    domain = sys.argv[1]
    alpha = int(sys.argv[2])
    try:
        prefix = sys.argv[3]
    except:
        prefix = ""
    run(os.path.join(BENCHMARK_DIR, domain), alpha, prefix)
