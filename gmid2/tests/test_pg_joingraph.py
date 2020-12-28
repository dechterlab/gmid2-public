import unittest
import os
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.basics.uai_files import read_mmap, read_vo, read_sum, read_mixed, write_vo
from gmid2.basics.undirected_network import PrimalGraph, call_variable_ordering
from gmid2.basics.directed_network import topo_sort, rev_topo_sort
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.bucket import bucket_tree_decomposition, mini_bucket_tree_decomposition, join_graph_decomposition
from gmid2.inference.pgm_gdd import PgmGDD
from gmid2.global_constants import *
from time import time


class TestJoinGraph(unittest.TestCase):
    def _test_small(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car.mixed")
        file_info = read_mixed(file_name, skip_table=False)      # only read uai file and fill in other information

        gm = GraphicalModel()
        gm.build(file_info)

        read_vo(file_name+".vo", file_info)
        elim_order = file_info.vo

        mbt = mini_bucket_tree_decomposition(gm, elim_order, ibound=1)
        jg = join_graph_decomposition(mbt, elim_order)
        print(jg)


    def test_BN(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "ID_from_BN_0_w28d6.mixed")
        file_info = read_mixed(file_name, skip_table=False)      # only read uai file and fill in other information

        gm = GraphicalModel()
        gm.build(file_info)

        read_vo(file_name+".vo", file_info)
        elim_order = file_info.vo

        mbt = mini_bucket_tree_decomposition(gm, elim_order, ibound=1)
        jg = join_graph_decomposition(mbt, elim_order)
        print(jg)

        scopes = {}
        for bids in topo_sort(jg):
            for bid in bids:
                bucket = jg.bid2bucket[bid]
                scopes[bid] = set(bucket.scope_vids)
                # print("{}:{}".format(bid, scopes[bid]))


        count = 0
        for i in scopes:
            for j in scopes:
                if i == j: continue

                if scopes[i] <= scopes[j]:
                    # print("{} {} {} {}".format(i, j, scopes[i], scopes[j]))
                    count += 1



