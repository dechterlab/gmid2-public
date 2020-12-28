import unittest
import os
from sortedcontainers import SortedSet, SortedDict, SortedList
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.basics.uai_files import read_limid, FileInfo, read_vo
from gmid2.basics.directed_network import DecisionNetwork
from gmid2.basics.undirected_network import PrimalGraph, greedy_variable_order
from gmid2.basics.undirected_network import get_induced_width_from_ordering, iterative_greedy_variable_order
from gmid2.global_constants import *


class PgCreationTest(unittest.TestCase):
    def setUp(self):
        self.file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        self.file_info = read_limid(self.file_name, skip_table=True)

    def test_creation_file_scope(self):
        print(self.id())
        pg = PrimalGraph()
        pg.build_from_scopes(self.file_info.scopes, self.file_info.var_types)

        # order of vid appear in scopes
        # nid   0 1 2 3 4 5
        # vid   1 2 0 3 4 5
        # type  C C D C C D
        self.assertEqual(['C', 'C', 'D', 'C', 'C', 'D'], [pg.nodes[nid]['type'] for nid in pg.nodes])
        self.assertEqual(SortedSet([0, 1, 3, 4]), pg.chance_nids)
        self.assertEqual(SortedSet([2, 5]), pg.decision_nids)
        self.assertEqual([2, 0, 1, 3, 4, 5], [pg.vid2nid[el] for el in range(6)])

        # edges
        edge_vid = [(1,0), (1,3), (0,3), (2, 0), (2, 4), (0,4), (1,5), (2,5)]
        src, dest = zip(*edge_vid)
        edge_nid = SortedList(zip([pg.vid2nid[el] for el in src], [pg.vid2nid[el] for el in dest]))
        self.assertEqual(edge_nid ,SortedList(pg.edges))
        print(edge_nid)
        print(SortedList(pg.edges))

    def test_creation_from_dn(self):
        print(self.id())
        dn = DecisionNetwork()
        dn.build(self.file_info)
        pg = PrimalGraph()
        pg.build_from_dn(dn)

        # print(list(dn.nodes[el]['type'] for el in dn.nodes))    # ['C', 'C', 'C', 'C', 'D', 'D', 'U', 'U', 'U']
        # print(list(dn.nodes[el]['vid'] for el in dn.nodes))     # [1, 2, 3, 4, 0, 5, None, None, None]
        # nid   0 1 2 3 4 5
        # vid   1 2 3 4 0 5
        # type  C C C C D D
        self.assertEqual(['C', 'C', 'C', 'C', 'D', 'D'], [pg.nodes[nid]['type'] for nid in pg.nodes])
        self.assertEqual(SortedSet([0, 1, 2, 3]), pg.chance_nids)
        self.assertEqual(SortedSet([4, 5]), pg.decision_nids)

        # edges
        edge_vid = [(1, 0), (1, 3), (0, 3), (2, 0), (2, 4), (0, 4), (1, 5), (2, 5)]
        src, dest = zip(*edge_vid)
        edge_nid = SortedList(zip([pg.vid2nid[el] for el in src], [pg.vid2nid[el] for el in dest]))
        edge_nid = SortedList(tuple(sorted(el)) for el in edge_nid)
        self.assertEqual(edge_nid ,SortedList(pg.edges))

    def test_ordering(self):
        print(self.id())
        pg = PrimalGraph()
        # order of vid appear in scopes
        # nid   0 1 2 3 4 5
        # vid   1 2 0 3 4 5
        # type  C C D C C D
        # [(0, 2), (0, 3), (0, 5), (1, 2), (1, 4), (1, 5), (2, 3), (2, 4)])
        pg.build_from_scopes(self.file_info.scopes, self.file_info.var_types)
        ordering, iw = greedy_variable_order(pg)
        # walk through the code
        print(ordering)     # [3, 4, 1, 0, 2, 5]
        print(iw)           # 2
        get_iw = get_induced_width_from_ordering(pg, ordering)
        self.assertEqual(2, get_iw)
        ordering, iw = iterative_greedy_variable_order(pg, 2)
        print(ordering, iw)

    def test_ordering_large(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "ID_from_BN_0_w33d11")
        file_info = read_limid(file_name, skip_table=True)
        file_name = file_name + ".vo"
        file_info = read_vo(file_name, file_info)

        pg = PrimalGraph()
        pg.build_from_scopes(file_info.scopes, file_info.var_types)
        pvo = []
        for each_block in file_info.blocks:     # blocks of constrained elim order
            pvo.append( [pg.vid2nid[vid] for vid in each_block] )

        ordering, iw = iterative_greedy_variable_order(primal_graph=pg, iter_limit=100, pvo=pvo, pool_size=16)
        print([pg.nid2vid[nid] for nid in ordering])
        print("found_iw:{}".format(iw))       # 22 too low?

        nid_ordering = [pg.vid2nid[vid] for vid in file_info.vo]
        get_iw = get_induced_width_from_ordering(pg, nid_ordering)
        print(file_info.vo)
        print("get_iw:{}".format(get_iw))      # 33 as shown
