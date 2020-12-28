import unittest
import os
import numpy.testing as nptest
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
import numpy as np
from gmid2.basics.uai_files import read_mmap, read_vo, read_sum, read_mpe
from gmid2.basics.undirected_network import PrimalGraph
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.bucket import bucket_tree_decomposition, mini_bucket_tree_decomposition, join_graph_decomposition
from gmid2.inference.pgm_bte import PgmBTE
from gmid2.inference.pgm_wmbmm import PgmWMBMM

import numpy.testing as nptest

from gmid2.basics.factor import *

class PgmWMBMMTSumest(unittest.TestCase):
    def setUp(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mmap"), "simple4.mmap")
        self.file_info = read_sum(file_name, skip_table=False)

        file_name2 = os.path.join(os.path.join(os.getcwd(), "test_data/bn"), "hailfinder")
        self.file_info2 = read_sum(file_name2, skip_table=False)
        read_vo(file_name2 + ".vo", self.file_info2)

    def _test_sum_bte(self):
        gm = GraphicalModel()
        gm.build(self.file_info)
        gm.convert_to_log()
        vid_elim_order = [5, 3, 1, 4, 2, 0]
        bt = bucket_tree_decomposition(gm, vid_elim_order)

        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(bt)
        bte.schedule(bt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=False)
        bound = bte.bounds()
        print("simple4 be:{}".format(bound))
        if gm.is_log:
            nptest.assert_almost_equal(-5.4840961933, bound)
        else:
            nptest.assert_almost_equal(0.004152286247993446, bound)

    def _tezst_sum_mbte(self):
        gm = GraphicalModel()
        gm.build(self.file_info)
        gm.convert_to_log()
        vid_elim_order = [5, 3, 1, 4, 2, 0]
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=2)

        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule(mbt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=False)
        bound = bte.bounds()
        print("simple4 mbe:{}".format(bound))
        if gm.is_log:
            self.assertGreaterEqual(bound, -5.4840961933)
        else:
            self.assertGreaterEqual(bound, 0.004152286247993446)
        # GMID
        # np.exp(-5.04508798328)
        # Out[45]: 0.006440893647970922

    def _test_sum_wmbmm(self):
        gm = GraphicalModel()
        gm.build(self.file_info)
        gm.convert_to_log()
        vid_elim_order = [5, 3, 1, 4, 2, 0]
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=2)

        bte = PgmWMBMM(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule()
        bte.init_propagate()
        bte.propagate_iter()
        bound = bte.bounds()
        print("simple4 wmbmm:{}".format(bound))
        if gm.is_log:
            self.assertGreaterEqual(bound, -5.4840961933)
        else:
            self.assertGreaterEqual(bound, 0.004152286247993446)

################## hailfinder

    def test_sum_bte_hailfinder(self):
        gm = GraphicalModel()
        gm.build(self.file_info2)
        gm.convert_to_log()
        vid_elim_order = self.file_info2.vo
        bt = bucket_tree_decomposition(gm, vid_elim_order)

        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(bt)
        bte.schedule(bt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=False)
        bound = bte.bounds()
        print("hailfinder be:{}".format(bound))
        if gm.is_log:
            nptest.assert_almost_equal(bound, 1.66533453694e-16, decimal=5)
        else:
            nptest.assert_almost_equal(bound, 1.0, decimal=5)
        # normalized to 1.0     PRECISION
        # 2.2300000224184657e-07
        # be: 1.0000002230000258

    def test_sum_mbe_hailfinder(self):
        gm = GraphicalModel()
        gm.build(self.file_info2)
        gm.convert_to_log()
        vid_elim_order = self.file_info2.vo
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=2)

        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule(mbt)
        bte.init_propagate()
        bte.propagate_iter()
        bound = bte.bounds()
        print("hailfinder mbe:{}".format(bound))

        if gm.is_log:
            self.assertGreaterEqual(bound, 0.0)
        else:
            self.assertGreaterEqual(bound, 1.0)


    def test_sum_wmbmm_hailfinder(self):
        gm = GraphicalModel()
        gm.build(self.file_info2)
        gm.convert_to_log()
        vid_elim_order = self.file_info2.vo
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=2)

        bte = PgmWMBMM(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule()
        bte.init_propagate()
        bte.propagate_iter()
        bound = bte.bounds()
        print("hailfinder wmbmm:{}".format(bound))

        if gm.is_log:
            self.assertGreaterEqual(bound, 0.0)
        else:

            self.assertGreaterEqual(bound, 1.0)

# wmbmm no notable improvement
# log scale be is zero ~ 1.0
# ibound 4
# hailfinder be:2.2300000224184657e-07
# hailfinder mbe:0.27565941354375034
# hailfinder wmbmm:0.27565941354375034
# ibound 2
# hailfinder be:2.2300000224184657e-07
# hailfinder mbe:4.288752518596624
# hailfinder wmbmm:4.288752518596624