import unittest
import os
from sortedcontainers import SortedSet, SortedDict, SortedList
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.global_constants import *
from gmid2.basics.uai_files import read_limid
from gmid2.basics.undirected_network import PrimalGraph, greedy_variable_order
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.bucket import bucket_tree_decomposition, mini_bucket_tree_decomposition, join_graph_decomposition

class BucketTreeTest(unittest.TestCase):
    def test_bucket_tree_decomposition(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=True)
        gm = GraphicalModel()
        gm.build(file_info)
        pg = PrimalGraph()
        pg.build_from_scopes(file_info.scopes, file_info.var_types)
        print(file_info.blocks)
        nid_pvo = [pg.vids2nids(each_block) for each_block in file_info.blocks]
        nid_ordering, iw = greedy_variable_order(pg, nid_pvo)
        vid_ordering = [pg.nid2vid[nid] for nid in nid_ordering]
        vid_elim_order = [1, 2, 5, 4, 3, 0]
        bt = bucket_tree_decomposition(gm, vid_elim_order)
        # [[1, 2], [5], [4, 3], [0]]
        #
        # BucketGraph
        # Bucket_0: (1, 0)[v:[0, 1, 3, 5], f:[0, 2, 5], m:[]]
        # Separator_0: [0, 2][v:[0, 3, 5]]
        #
        # Bucket_1: (2, 0)[v:[0, 2, 4, 5], f:[1, 3, 6], m:[]]
        # Separator_1: [1, 2][v:[0, 4, 5]]
        #
        # Bucket_2: (5, 0)[v:[0, 3, 4, 5], f:[], m:[0, 1]]
        # Separator_2: [2, 3][v:[0, 3, 4]]
        #
        # Bucket_3: (4, 0)[v:[0, 3, 4], f:[], m:[2]]
        # Separator_3: [3, 4][v:[0, 3]]
        #
        # Bucket_4: (3, 0)[v:[0, 3], f:[], m:[3]]
        # Separator_4: [4, 5][v:[0]]
        #
        # Bucket_5: (0, 0)[v:[0], f:[4], m:[4]]
        print(bt)


    def test_mini_bucket_tree_decomposition(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=True)
        gm = GraphicalModel()
        gm.build(file_info)
        vid_elim_order = [1, 2, 5, 4, 3, 0]
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, 2)
        # BucketGraph ibound 2
        # Bucket_0: (1, 0)[v:[1, 5], f:[0, 5], m:[]]
        # Separator_0: [0, 4][v:[5]]
        # ConsistencyConstraint_2: [0, 1][v:[1]]
        #
        # Bucket_2: (2, 0)[v:[2, 5], f:[1, 6], m:[]]
        # Separator_3: [2, 4][v:[5]]
        # ConsistencyConstraint_5: [2, 3][v:[2]]
        #
        # Bucket_1: (1, 1)[v:[0, 1, 3], f:[2], m:[2]]
        # Separator_1: [1, 6][v:[0, 3]]
        #
        # Bucket_3: (2, 1)[v:[0, 2, 4], f:[3], m:[5]]
        # Separator_4: [3, 5][v:[0, 4]]
        #
        # Bucket_4: (5, 0)[v:[5], f:[], m:[0, 3]]
        #
        # Bucket_5: (4, 0)[v:[0, 4], f:[], m:[4]]
        # Separator_6: [5, 7][v:[0]]
        #
        # Bucket_6: (3, 0)[v:[0, 3], f:[], m:[1]]
        # Separator_7: [6, 7][v:[0]]
        #
        # Bucket_7: (0, 0)[v:[0], f:[4], m:[6, 7]]
        print(mbt)

    def test_join_graph_decomposition(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=True)
        gm = GraphicalModel()
        gm.build(file_info)
        vid_elim_order = [1, 2, 5, 4, 3, 0]
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, 2)
        jg = join_graph_decomposition(mbt , vid_elim_order)

        # BucketGraph   ibound 2
        # Bucket_0: (1, 0)[v:[1, 5], f:[0, 5], m:[]]
        # Separator_0: [0, 2][v:[5]]
        # Separator_2: [0, 1][v:[1]]
        #
        # Bucket_2: (2, 0)[v:[2, 5], f:[1, 6], m:[0]]
        # Separator_5: [2, 3][v:[2]]
        #
        # Bucket_3: (2, 1)[v:[0, 2, 4], f:[3], m:[5]]
        # Separator_6: [3, 1][v:[0]]
        #
        # Bucket_1: (1, 1)[v:[0, 1, 3], f:[2], m:[2, 6]]
        print(jg)

    def test_join_graph_decomposition_tree(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=True)
        gm = GraphicalModel()
        gm.build(file_info)
        vid_elim_order = [1, 2, 5, 4, 3, 0]
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, 10)
        jg = join_graph_decomposition(mbt , vid_elim_order)
        print(jg)
