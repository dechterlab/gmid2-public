import unittest
import os
import numpy.testing as nptest
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.basics.uai_files import read_mmap, read_vo, read_sum, write_vo, read_mixed
from gmid2.basics.undirected_network import PrimalGraph, iterative_greedy_variable_order
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.bucket import bucket_tree_decomposition, mini_bucket_tree_decomposition, join_graph_decomposition
from gmid2.inference.pgm_bte import PgmBTE


class PgmBteForMiniBucketTest(unittest.TestCase):
    def test_run_pgm_mbte_as_sum(self):
        print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mmap"), "simple4.mmap")
        file_info = read_sum(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        pg = PrimalGraph()
        pg.build_from_scopes(file_info.scopes, file_info.var_types)
        read_vo(file_name + ".vo", file_info)
        vid_elim_order = [5, 3, 1, 4, 2, 0]     #file_info.vo   # 5 3 1 4 2 0
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=2)
        # print(mbt)

        # Message Passing
        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule(mbt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=True)
        bound = bte.bounds()
        # print(bound)
        # for cid in bte:
        #     print("{}:{}\n".format(cid, bte.bounds_at(cid)))
        nptest.assert_almost_equal(bound, 0.03268429679459796)

    def test_run_pgm_mbte_as_mmap(self):
        print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mmap"), "simple4.mmap")
        file_info = read_mmap(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        pg = PrimalGraph()
        pg.build_from_scopes(file_info.scopes, file_info.var_types)
        read_vo(file_name + ".vo", file_info)
        vid_elim_order = [5, 3, 1, 4, 2, 0]  # file_info.vo   # 5 3 1 4 2 0
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=2)
        # print(mbt)

        # Message Passing
        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule(mbt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=False)
        bound = bte.bounds()
        # print(bound)
        nptest.assert_almost_equal(bound, 0.012129264312566184)

    def test_run_pgm_mbte_jg_as_mmap(self):
        print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mmap"), "simple4.mmap")
        file_info = read_mmap(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        pg = PrimalGraph()
        pg.build_from_scopes(file_info.scopes, file_info.var_types)
        read_vo(file_name + ".vo", file_info)
        vid_elim_order = [5, 3, 1, 4, 2, 0]  # file_info.vo   # 5 3 1 4 2 0
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=2)
        mbt = join_graph_decomposition(mbt, vid_elim_order)
        # print(mbt)

        # Message Passing
        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule(mbt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=False)
        bound = bte.bounds()
        # print(bound)
        nptest.assert_almost_equal(bound, 0.012129264312566184)

    def test_run_pgm_mbte_BN_0_as_mmap(self):
        print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mmap"), "ID_from_BN_0_w28d6.mmap")
        file_info = read_mmap(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm = GraphicalModel()
        gm.build(file_info)
        pg = PrimalGraph()
        pg.build_from_scopes(file_info.scopes, file_info.var_types)

        pvo = []
        for each_block in file_info.blocks:  # blocks of constrained elim order
            pvo.append([pg.vid2nid[vid] for vid in each_block])

        # vid_elim_order, iw = iterative_greedy_variable_order(pg, 1000, pvo, pool_size=20)
        # print("iw:{}".format(iw))
        # write_vo(file_name + ".vo", vid_elim_order, iw)

        read_vo(file_name+".vo", file_info)
        vid_elim_order = file_info.vo

        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=20)
        # mbt = join_graph_decomposition(mbt, vid_elim_order)
        # print(mbt)

        # Message Passing
        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule(mbt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=False)
        bound = bte.bounds()

        print("as mmap iw 38 i 20 : {}".format(bound))
        # iw: 38
        # 6748.19268

        # GMID random paritions
        # 831603325
        # 71532180

    def test_run_pgm_mbte_BN_0_as_mixed(self):
        print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mixed"), "ID_from_BN_0_w28d6.mixed")
        file_info = read_mixed(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm = GraphicalModel()
        gm.build(file_info)

        gm.convert_to_log()
        pg = PrimalGraph()
        pg.build_from_scopes(file_info.scopes, file_info.var_types)


        # pvo = []
        # for each_block in file_info.blocks:  # blocks of constrained elim order
        #     pvo.append([pg.vid2nid[vid] for vid in each_block])
        # vid_elim_order, iw = iterative_greedy_variable_order(pg, 1000, pvo, pool_size=20)
        # print("iw:{}".format(iw))
        # write_vo(file_name + ".vo", vid_elim_order, iw)

        read_vo(file_name+".vo", file_info)
        vid_elim_order = file_info.vo

        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=20)
        # mbt = join_graph_decomposition(mbt, vid_elim_order)       # cannot use JG here!
        # print(mbt)

        # Message Passing
        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule(mbt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=False)
        bound = bte.bounds()

        print("as mixed i 20 : {}".format(bound))

        # 14899412305478
        # 30.332342885516752    log

        # GMID log true
        # 39357
        # 84891