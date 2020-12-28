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
        self.file_name = "simple4.mmap"
        # self.file_name = "hailfinder"
        # v1 = Variable(100, 2, 'C')
        # f1 = Factor([v1], 2.0)
        # print(f1)

    def _test_sum_bte(self):
        # print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mmap"), self.file_name)
        file_info = read_mmap(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_to_log()

        read_vo(file_name + ".vo", file_info)
        vid_elim_order = [5, 3, 1, 4, 2, 0]     #file_info.vo   # 5 3 1 4 2 0     simple4
        bt = bucket_tree_decomposition(gm, vid_elim_order)

        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(bt)
        bte.schedule(bt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=False)
        bound = bte.bounds()
        print("be:{}".format(bound))

        if gm.is_log:
            nptest.assert_almost_equal(-6.72879777427, bound)
        else:
            nptest.assert_almost_equal(0.00119596992, bound)


    def test_sum_mbte(self):
        # print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mmap"), self.file_name)
        file_info = read_mmap(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_to_log()

        read_vo(file_name + ".vo", file_info)
        vid_elim_order = [5, 3, 1, 4, 2, 0]     #file_info.vo   # 5 3 1 4 2 0
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=2)

        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule(mbt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=False)
        bound = bte.bounds()
        print("mbe:{}".format(bound))

        if gm.is_log:
            self.assertGreaterEqual(bound, -6.72879777427)      # a >= b
        else:
            self.assertGreaterEqual(bound, 0.00119596992)
        # GMID differnt paritioning
        # 0.00266450688
        # -5.74007135926

    def test_sum_wmbmm(self):
        # print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mmap"), self.file_name)
        file_info = read_mmap(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_to_log()

        read_vo(file_name + ".vo", file_info)
        vid_elim_order = file_info.vo
        # vid_elim_order = [5, 3, 1, 4, 2, 0]     #file_info.vo   # 5 3 1 4 2 0
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=2)

        bte = PgmWMBMM(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule()
        bte.init_propagate()
        bte.propagate_iter()
        bound = bte.bounds()
        print("wmbmm:{}".format(bound))
        # no WMBMM implemntation in GMID; higher than exact; but lower than mbe?
        # wmbmm: -6.019765162382094
        if gm.is_log:
            self.assertGreaterEqual(bound, -6.72879777427)      # a >= b
        else:
            self.assertGreaterEqual(bound, 0.00119596992)

