PRJ_PATH = "/home/junkyul/conda2/gmid2"
import sys
sys.path.append(PRJ_PATH)

import os
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.basics.uai_files import read_mmap, read_vo, read_sum, read_mixed, write_vo
from gmid2.basics.undirected_network import PrimalGraph, call_variable_ordering
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.bucket import bucket_tree_decomposition, mini_bucket_tree_decomposition, join_graph_decomposition
from gmid2.inference.pgm_gdd import PgmGDD
from gmid2.global_constants import *
from time import time

# runs but very slow!
def test_gdd_1():
    file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "rand-c20d2o1-01.mixed")
    file_info = read_mixed(file_name, skip_table=False)      # only read uai file and fill in other information

    gm = GraphicalModel()
    gm.build(file_info)
    gm.convert_to_log()

    read_vo(file_name+".vo", file_info)
    elim_order = file_info.vo

    mbt = mini_bucket_tree_decomposition(gm, elim_order, ibound=1)
    jg = join_graph_decomposition(mbt, elim_order)

    # Message Passing
    iter_options = { 'iter_limit':1000 }
    opt_options = { 'ls_steps': 30, 'armijo_thr': 1e-4, 'armijo_step_back': 0.5,
                    'ls_tolerance':TOL, 'gd_steps':20, 'tolerance': TOL}
    gdd = PgmGDD(gm, elim_order, iter_options, opt_options)
    gdd.build_message_graph(jg)
    gdd.schedule()
    gdd.init_propagate()
    t0 = time()
    gdd.propagate_iter()
    bound = gdd.bounds()
    print("{}\tiw\t6\ti\t1\tgdd:{}\t(opt 3.695057019)\t(JGDID i=1 4.366097966)".format(time()-t0, bound))
    # 249.86 after 147 sec GMID
    # 234 after 3876 sec

# 999	686.0910284519196	0.6745457649230957	4.967287078549987e-06	234.63698350761365
# 1000	686.7788982391357	0.6878418922424316	5.01299448529835e-06	234.6358072766575


if __name__ == "__main__":
    test_gdd_1()

