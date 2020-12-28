import unittest
import os
import numpy.testing as nptest
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.basics.uai_files import read_mmap, read_vo, read_sum, read_mixed
from gmid2.basics.undirected_network import PrimalGraph
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.bucket import bucket_tree_decomposition, mini_bucket_tree_decomposition, join_graph_decomposition
from gmid2.inference.pgm_gdd import PgmGDD
from gmid2.global_constants import *

class PgmGDDCarMixedoverMBTree(unittest.TestCase):
    def setUp(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car.mixed")
        self.file_info = read_mixed(file_name, skip_table=False)      # only read uai file and fill in other information
        self.elim_order = [6, 2, 1, 5, 4, 3, 0]    # any other is OK just flatten pvo

    def test_gdd_mbtree_1(self):
        gm = GraphicalModel()
        gm.build(self.file_info)
        gm.convert_to_log()

        mbt = mini_bucket_tree_decomposition(gm, self.elim_order, ibound=10)

        # Message Passing
        iter_options = { 'iter_limit':1000 }
        opt_options = { 'ls_steps': 30, 'armijo_thr': 1e-4, 'armijo_step_back': 0.5,
                        'ls_tolerance':TOL, 'gd_steps':20, 'tolerance': TOL}
        gdd = PgmGDD(gm, self.elim_order, iter_options, opt_options)
        gdd.build_message_graph(mbt)
        gdd.schedule()
        gdd.init_propagate()
        gdd.propagate_iter()
        bound = gdd.bounds()
        print("gdd i:{} (opt 70)".format(10, bound))

        # this is slow, and terminate earlier than join graph
        #

    def test_gdd_mbtree_2(self):
        gm = GraphicalModel()
        gm.build(self.file_info)
        gm.convert_to_log()

        mbt = mini_bucket_tree_decomposition(gm, self.elim_order, ibound=1)

        # Message Passing
        iter_options = { 'iter_limit':1000 }
        opt_options = { 'ls_steps': 30, 'armijo_thr': 1e-4, 'armijo_step_back': 0.5,
                        'ls_tolerance':TOL, 'gd_steps':20, 'tolerance': TOL}
        gdd = PgmGDD(gm, self.elim_order, iter_options, opt_options)
        gdd.build_message_graph(mbt)
        gdd.schedule()
        gdd.init_propagate()
        gdd.propagate_iter()
        bound = gdd.bounds()
        print("gdd i:{} (opt 70)".format(1, bound))



