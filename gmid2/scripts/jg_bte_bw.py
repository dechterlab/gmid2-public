PRJ_PATH = "/home/junkyul/conda/gmid2"
import sys
sys.path.append(PRJ_PATH)

import os
import time
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.global_constants import *
from gmid2.basics.uai_files import read_limid, read_vo, write_vo
from gmid2.basics.undirected_network import PrimalGraph, call_variable_ordering, get_induced_width_from_ordering
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.bucket import bucket_tree_decomposition
from gmid2.inference.pgm_bte import PgmBTELog


def run(file_path):
    print("{}".format(PgmBTELog.__name__))
    f = file_path.split("/")[-1].replace(".uai", "")
    print("\nSTART {}\t\t{}".format(f, time.ctime(time.time())))
    file_name = file_path.replace(".uai", "")
    file_info = read_limid(file_name, skip_table=False)
    gm = GraphicalModel()
    gm.build(file_info)
    gm.convert_prob_to_log()
    gm.convert_util_to_alpha(1.0)

    t0 = time.time()
    pg = PrimalGraph()
    pg.build_from_scopes(gm.scope_vids)
    try:
        print("read vo from file")
        read_vo(file_name + ".vo", file_info)
        bw_ordering = file_info.vo
        bw_iw = get_induced_width_from_ordering(pg, nid_ordering=[pg.nid2vid[el] for el in bw_ordering])
    except:
        pvo_vid = file_info.blocks
        bw_ordering, bw_iw = call_variable_ordering(pg, 10000, pvo_vid)
        write_vo(file_name + ".vo", bw_ordering, bw_iw)
    print("w\t\t{}\nvo\t\t{}".format(bw_iw, " ".join(str(el) for el in bw_ordering)))
    bt = bucket_tree_decomposition(gm, bw_ordering)
    bte = PgmBTELog(gm, bw_ordering)
    bte.build_message_graph(bt)
    print("build\t\t{}".format(time.time() - t0))
    bte.schedule(bt)
    bte.init_propagate()
    t0 = time.time()
    bte.propagate_iter()
    bound = bte.bounds()
    print("prop\t\t{}".format(time.time() - t0))
    print("ub\t\t{}".format(bound))
    print("END {}\t\t{}".format(f, time.ctime(time.time())))
    return bound


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        TEST_PATH = os.path.join(BENCHMARK_DIR, "synthetic")
        f = "mdp1-4_2_2_5.uai"
        file_path = os.path.join(TEST_PATH, f)

    run(file_path)