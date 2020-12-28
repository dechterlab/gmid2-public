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


def test_gdd_2():
    file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "rand-c30d3o1-01.mixed")
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
    print("{}\t{}".format(t0-time(), np.exp(bound)))
    # 1217 after more 7000 sec GMID
    # 1299 after 360 sec GMID

# 999	1662.5867567062378	1.3839316368103027	0.00041943243686670684	1282.7671591849914
# 1000	1663.986073732376	1.3992886543273926	8.375625671686038e-05	1282.6597239087498

if __name__ == "__main__":
    test_gdd_2()

