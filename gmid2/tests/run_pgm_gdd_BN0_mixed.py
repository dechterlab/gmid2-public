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

def test_gdd_1():
    file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "ID_from_BN_0_w28d6.mixed")
    file_info = read_mixed(file_name, skip_table=False)      # only read uai file and fill in other information

    gm = GraphicalModel()
    gm.build(file_info)
    gm.convert_to_log()

    # pg = PrimalGraph()
    # pg.build_from_scopes(scopes=file_info.scopes, var_types=file_info.var_types)
    # pvo_vid = []
    # for block in file_info.blocks:
    #     pvo_vid.append([pg.vid2nid[vid] for vid in block])
    # elim_order, iw = call_variable_ordering(pg, iter_limit=5, pvo_vid=pvo_vid)
    # write_vo(file_name + ".vo", elim_order, iw)
    read_vo(file_name+".vo", file_info)
    elim_order = file_info.vo

    mbt = mini_bucket_tree_decomposition(gm, elim_order, ibound=1)
    jg = mbt

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
    print("{}\tiw\t30\ti\t1\tgdd:{}\t(JGDID i=1 53.139311958)".format(time()-t0, np.exp(bound)))

def test_gdd_2():
    file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "ID_from_BN_0_w28d6.mixed")
    file_info = read_mixed(file_name, skip_table=False)      # only read uai file and fill in other information

    gm = GraphicalModel()
    gm.build(file_info)
    gm.convert_to_log()

    # pg = PrimalGraph()
    # pg.build_from_scopes(scopes=file_info.scopes, var_types=file_info.var_types)
    # pvo_vid = []
    # for block in file_info.blocks:
    #     pvo_vid.append([pg.vid2nid[vid] for vid in block])
    # elim_order, iw = call_variable_ordering(pg, iter_limit=5, pvo_vid=pvo_vid)
    # write_vo(file_name + ".vo", elim_order, iw)
    read_vo(file_name+".vo", file_info)
    elim_order = file_info.vo

    mbt = mini_bucket_tree_decomposition(gm, elim_order, ibound=1)
    jg = join_graph_decomposition(mbt, elim_order)

    # Message Passing
    iter_options = { 'iter_limit':100 }
    opt_options = { 'ls_steps': 30, 'armijo_thr': 1e-4, 'armijo_step_back': 0.5,
                    'ls_tolerance':TOL, 'gd_steps':20, 'tolerance': TOL}
    gdd = PgmGDD(gm, elim_order, iter_options, opt_options)
    gdd.build_message_graph(jg)
    gdd.schedule()
    gdd.init_propagate()
    t0 = time()
    gdd.propagate_iter()
    bound = gdd.bounds()
    print("{}\tiw\t30\ti\t1\tgdd:{}\t(JGDID i=1 53.139311958)".format(time()-t0, np.exp(bound)))


if __name__ == "__main__":
    test_gdd_1()
    # test_gdd_2()


