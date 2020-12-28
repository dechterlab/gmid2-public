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


class PgmWMBMMTest(unittest.TestCase):
    def setUp(self):
        self.file_name = "simple4.mmap"

    def test_mpe_bte(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mmap"), self.file_name)
        file_info = read_mpe(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_to_log()

        vid_elim_order = [5, 3, 1, 4, 2, 0]
        bt = bucket_tree_decomposition(gm, vid_elim_order)

        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(bt)
        bte.schedule(bt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=False)
        bound = bte.bounds()
        print("be:{}".format(bound))
        if gm.is_log:
            nptest.assert_almost_equal(-7.26424224059, bound)
        else:
            nptest.assert_almost_equal(0.0007001316, bound)

    def test_mpe_mbte(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mmap"), self.file_name)
        file_info = read_mpe(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_to_log()
        vid_elim_order = [5, 3, 1, 4, 2, 0]
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=2)

        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule(mbt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=False)
        bound = bte.bounds()
        print("mbe:{}".format(bound))
        if gm.is_log:
            self.assertGreaterEqual(bound, -7.26424224059)      # a >= b
        else:
            self.assertGreaterEqual(bound, 0.0007001316)
        # GMID
        # np.exp(-6.77849025438)
        # Out[43]: 0.0011379916799963354


    def test_mpe_wmbmm(self):
        # print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mmap"), self.file_name)
        file_info = read_mpe(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_to_log()

        vid_elim_order = [5, 3, 1, 4, 2, 0]
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=2)

        bte = PgmWMBMM(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule()
        bte.init_propagate()
        bte.propagate_iter()
        bound = bte.bounds()
        print("wmbmm:{}".format(bound))

        if gm.is_log:
            self.assertGreaterEqual(bound, -7.26424224059)      # a >= b
        else:
            self.assertGreaterEqual(bound, 0.0007001316)
        # wmbmm: -6.729021511644712
        # no implementation in GMID

