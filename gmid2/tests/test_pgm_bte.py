import unittest
import os
import numpy.testing as nptest
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.basics.uai_files import read_mmap, read_vo, read_sum
from gmid2.basics.undirected_network import PrimalGraph
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.bucket import bucket_tree_decomposition, mini_bucket_tree_decomposition, join_graph_decomposition
from gmid2.inference.pgm_bte import PgmBTE


class PgmBteTest(unittest.TestCase):
    def test_run_pgm_bte_mmap(self):
        print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "simple4.mmap")
        file_info = read_mmap(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        pg = PrimalGraph()
        pg.build_from_scopes(file_info.scopes, file_info.var_types)
        file_info = read_vo(file_name + ".vo", file_info)
        vid_elim_order = file_info.vo   # 3 2 4 0 1
        bt = bucket_tree_decomposition(gm, vid_elim_order)

        # Message Passing
        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(bt)
        bte.schedule(bt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=False)
        bound = bte.bounds()
        for cid in bte:
            print("{}:{}\n".format(cid, bte.bounds_at(cid)))
        nptest.assert_almost_equal(bound, 0.00119596992)
        # name: simple4.mmap
        # final
        # MEU: 0.00119596992

    def test_run_pgm_bte_as_summation(self):
        print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "simple4.mmap")
        file_info = read_sum(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        pg = PrimalGraph()
        pg.build_from_scopes(file_info.scopes, file_info.var_types)
        file_info = read_vo(file_name + ".vo", file_info)
        vid_elim_order = file_info.vo   # 3 2 4 0 1
        bt = bucket_tree_decomposition(gm, vid_elim_order)
        print(bt)
        # BucketGraph
        # Bucket_0: (5, 0)[v:[1, 2, 4, 5], f:[7, 9, 11], m:[]]
        # Separator_0: [0, 2][v:[1, 2, 4]]
        #
        # Bucket_1: (3, 0)[v:[0, 1, 3, 4], f:[2, 5, 10], m:[]]
        # Separator_1: [1, 2][v:[0, 1, 4]]
        #
        # Bucket_2: (1, 0)[v:[0, 1, 2, 4], f:[0, 4, 6], m:[0, 1]]
        # Separator_2: [2, 3][v:[0, 2, 4]]
        #
        # Bucket_3: (4, 0)[v:[0, 2, 4], f:[3, 8], m:[2]]
        # Separator_3: [3, 4][v:[0, 2]]
        #
        # Bucket_4: (2, 0)[v:[0, 2], f:[1], m:[3]]
        # Separator_4: [4, 5][v:[0]]
        #
        # Bucket_5: (0, 0)[v:[0], f:[], m:[4]]

        # Message Passing
        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(bt)
        bte.schedule(bt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=True)
        bound = bte.bounds()
        for cid in bte:
            print("{}:{}\n".format(cid, bte.bounds_at(cid)))
        nptest.assert_almost_equal(bound, 0.00415228624)
        # START
        # name: simple4.sum
        # final
        # MEU: 0.004152286248
        # END

    def test_run_cte_summation(self):
        print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "simple4.mmap")
        file_info = read_sum(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        pg = PrimalGraph()
        pg.build_from_scopes(file_info.scopes, file_info.var_types)
        file_info = read_vo(file_name + ".vo", file_info)
        vid_elim_order = file_info.vo  # 3 2 4 0 1
        bt = bucket_tree_decomposition(gm, vid_elim_order)
        bt = join_graph_decomposition(bt, vid_elim_order)
        print(bt)
        # BucketGraph
        # Bucket_0: (5, 0)[v:[1, 2, 4, 5], f:[7, 9, 11], m:[]]
        # Separator_0: [0, 2][v:[1, 2, 4]]
        #
        # Bucket_1: (3, 0)[v:[0, 1, 3, 4], f:[2, 5, 10], m:[]]
        # Separator_1: [1, 2][v:[0, 1, 4]]
        #
        # Bucket_2: (1, 0)[v:[0, 1, 2, 4], f:[0, 4, 6], m:[0, 1]]

        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(bt)
        bte.schedule(bt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=True)
        bound = bte.bounds()
        for cid in bte:
            print("{}:{}\n".format(cid, bte.bounds_at(cid)))
        nptest.assert_almost_equal(bound, 0.00415228624)

# Arrays are not almost equal to 7 decimals
#  ACTUAL: 0.07202304147901326
#  DESIRED: 0.00415228624