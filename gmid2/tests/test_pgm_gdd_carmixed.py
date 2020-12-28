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

class PgmGddCarMixed(unittest.TestCase):
    def setUp(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car.mixed")
        self.file_info = read_mixed(file_name, skip_table=False)      # only read uai file and fill in other information
        self.elim_order = [6, 2, 1, 5, 4, 3, 0]    # any other is OK just flatten pvo

    def test_gdd_carmixed_1(self):
        gm = GraphicalModel()
        gm.build(self.file_info)
        gm.convert_to_log()

        mbt = mini_bucket_tree_decomposition(gm, self.elim_order, ibound=10)
        jg = join_graph_decomposition(mbt, self.elim_order)

        # Message Passing
        iter_options = { 'iter_limit':1000 }
        opt_options = { 'ls_steps': 30, 'armijo_thr': 1e-4, 'armijo_step_back': 0.5,
                        'ls_tolerance':TOL, 'gd_steps':20, 'tolerance': TOL}
        gdd = PgmGDD(gm, self.elim_order, iter_options, opt_options)
        gdd.build_message_graph(jg)
        gdd.schedule()
        gdd.init_propagate()
        gdd.propagate_iter()
        bound = gdd.bounds()
        print("gdd i:{} (opt 70)".format(10, bound))
# 820	167.5190110206604	0.14445972442626953	9.559419922311463e-10	70.15680658989454
# 821	167.66754364967346	0.14841389656066895	0.0	70.15680658989454


    def test_gdd_carmixed_2(self):
        gm = GraphicalModel()
        gm.build(self.file_info)
        gm.convert_to_log()

        mbt = mini_bucket_tree_decomposition(gm, self.elim_order, ibound=3)
        jg = join_graph_decomposition(mbt, self.elim_order)

        # Message Passing
        iter_options = { 'iter_limit':1000 }
        opt_options = { 'ls_steps': 30, 'armijo_thr': 1e-4, 'armijo_step_back': 0.5,
                        'ls_tolerance':TOL, 'gd_steps':20, 'tolerance': TOL}
        gdd = PgmGDD(gm, self.elim_order, iter_options, opt_options)
        gdd.build_message_graph(jg)
        gdd.schedule()
        gdd.init_propagate()
        gdd.propagate_iter()
        bound = gdd.bounds()
        print("gdd 3:{} (opt 70)".format(1, bound))
# i =1
# 999	109.53076100349426	0.10300326347351074	6.682476918484781e-07	1581.554991466176
# 1000	109.63955068588257	0.10863971710205078	6.66889651057545e-07	1581.5539367438712

# i=3
# 393	175.53950190544128	0.3451716899871826	2.671337817616859e-09	71.97637468594843
# 394	175.89632964134216	0.3567047119140625	0.0	71.97637468594843
