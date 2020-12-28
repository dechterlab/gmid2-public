import unittest
import os
from gmid2.basics.uai_files import read_limid
from gmid2.basics.directed_network import *
from sortedcontainers import SortedSet, SortedDict, SortedList
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


class DnFunctionTest(unittest.TestCase):
    def setUp(self):
        self.file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        self.file_info = read_limid(self.file_name, skip_table=False)
        self.dn = DecisionNetwork()
        self.dn.build(self.file_info)

    def test_parents(self):
        print("\n\n{}".format(self.id()))
        p1 = parents(self.dn, 0)        # pa of 0 is empty
        self.assertEqual(p1, SortedSet([]), "p1 failed")

        p2 = parents(self.dn, (2,3))   # pa of 2: 0, 4  pa of 3: 1, 4
        self.assertEqual(p2, SortedSet([0, 4, 1, 4]), "p2 failed")

        p3 = parents(self.dn, SortedSet( [7, 8] ))
        self.assertEqual(p3, SortedSet([0, 5, 1, 5]), "p3 failed")

        p4 = parents(self.dn, SortedSet([2, 3, 8]))
        self.assertEqual(p4, SortedSet([0, 4, 4, 1, 1, 5]), "p4 failed")

    def test_children(self):
        print("\n\n{}".format(self.id()))
        c1 = children(self.dn, 0)        # pa of 0 is empty
        self.assertEqual(c1, SortedSet([2, 7]), "c1 failed")

        c2 = children(self.dn, (2,3))   # pa of 2: 0, 4  pa of 3: 1, 4
        self.assertEqual(c2, SortedSet([5, 5]), "c2 failed")

        c3 = children(self.dn, SortedSet( [7, 8] ))
        self.assertEqual(c3, SortedSet([]), "c3 failed")

        c4 = children(self.dn, SortedSet([2, 3, 8]))
        self.assertEqual(c4, SortedSet([5, 5]), "c4 failed")

    # def test_neighbors(self):
    #     print("\n\n{}".format(self.id()))
    #     G = self.dn.to_undirected()
    #     n1 = neighbors(G, 0)
    #     self.assertEqual(n1, SortedSet([2, 7]))
    #
    #     n2 = neighbors(G, (2, 8))
    #     self.assertEqual(n2, SortedSet([0, 4, 5, 1, 5]))
    #
    #     n3 = neighbors(G, [4, 7])
    #     self.assertEqual(n3, SortedSet([6, 2, 3, 0, 5]))
    #
    #     n4 = neighbors(G, SortedSet([6, 7, 8]))
    #     self.assertEqual(n4, SortedSet([4, 0, 5, 1, 5]))

    def test_descendants(self):
        print("\n\n{}".format(self.id()))
        d1 = descendants(self.dn, 4)
        self.assertEqual(d1, SortedSet([2, 3, 5, 6, 7, 8]), "d1 failed")

        d2 = descendants(self.dn, [0, 5])
        self.assertEqual(d2, SortedSet([7, 8, 2, 7, 5]), "d2 failed")

        d3 = descendants(self.dn, SortedSet([0, 5]))
        self.assertEqual(d3, SortedSet([7, 8, 2, 7, 5]), "d3 failed")

    def test_ancestors(self):
        print("\n\n{}".format(self.id()))
        a1 = ancestors(self.dn, 4)
        self.assertEqual(a1, SortedSet([]), "a1 failed")

        a2 = ancestors(self.dn, [2, 8])
        self.assertEqual(a2, SortedSet([0, 4, 1, 5, 3, 2, 4]))

        a3 = ancestors(self.dn, SortedSet([6]))
        self.assertEqual(a3, SortedSet([4]))

    def test_markovblanket(self):
        print("\n\n{}".format(self.id()))
        m1 = markov_blanket_directed(self.dn, 2)
        self.assertEqual(m1, SortedSet([0, 4, 5, 3]))

        m2 = markov_blanket_directed(self.dn, [2, 3])
        self.assertEqual(m2, SortedSet([0, 4, 5, 4, 1, 5]))

    def test_rev_topo_sort(self):
        print("\n\n{}".format(self.id()))
        o1= list(rev_topo_sort(self.dn))
        self.assertEqual(o1, [SortedSet([6, 7, 8]),  SortedSet([5]), SortedSet([2, 3]), SortedSet([0, 1, 4]) ] )

    def test_topo_sort(self):
        print("\n\n{}".format(self.id()))
        o1= list(topo_sort(self.dn))
        self.assertEqual(o1, [SortedSet([0, 4, 1]), SortedSet([6, 2, 3]), SortedSet([5]), SortedSet([7, 8])])

    def test_reduce_nids(self):
        print("\n\n{}".format(self.id()))
        o1 = list(filter_subsets(rev_topo_sort(self.dn), self.dn.decision_nids))
        self.assertEqual(o1, [SortedSet([5]), SortedSet([4])] )

        o2 = list(filter_subsets(topo_sort(self.dn), self.dn.decision_nids))
        self.assertEqual(o2, [SortedSet([4]), SortedSet([5])] )

    def test_dsep(self):
        print("\n\n{}".format(self.id()))
        d = DecisionNetwork()
        d.build(self.file_info)
        r1 = reachble(d, x= 4, Z=SortedSet([1, 2, 3, 7]))
        self.assertEqual(r1, SortedSet([0, 4, 6, 5, 8]), "r1 failed")

        r2 = reachble(d, x= 0, Z=SortedSet([2, 3]))
        self.assertEqual(r2, SortedSet([0, 7, 4, 6, 1, 8, 5]), "r2 failed")

        r3 = reachble(d, x=0, Z=SortedSet([1, 2, 3, 7]))
        self.assertEqual(r3, SortedSet([0, 4, 6, 5, 8]), "r3 failed")

        self.assertTrue(dconnected(d, X=0, Y=5, Z=SortedSet([1, 2, 3, 7])))

        self.assertTrue(dconnected(d, X=SortedSet([0, 4]), Y=SortedSet([5, 8]), Z=SortedSet([1, 2, 3, 7])))

    def test_barern_node_removal(self):
        print("\n\n{}".format(self.id()))
        d = DecisionNetwork()
        d.build(self.file_info)
        nid1 = d.add_node_by_nid(type=TYPE_CHANCE_NODE, vid=6, fid=7)
        nid2 = d.add_node_by_nid(type=TYPE_CHANCE_NODE, vid=7, fid=8)
        d.add_edges_by_nid(nid=nid1, parent_nids=[0, 2])
        d.add_edges_by_nid(nid=nid2, parent_nids=[1, 3])
        barren = barren_nodes(d, probs=d.chance_nids, evids=d.value_nids)
        self.assertEqual(barren, SortedSet([nid1, nid2]))