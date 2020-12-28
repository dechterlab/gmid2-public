import unittest
import os
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


from gmid2.global_constants import *
from gmid2.basics.uai_files import read_limid
from gmid2.basics.helper import filter_subsets
from gmid2.basics.directed_network import *
from gmid2.inference.submodel import *


class SubmodelTreeTestID(unittest.TestCase):
    def setUp(self):
        self.file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "jensen")
        self.file_info = read_limid(self.file_name, skip_table=False)
        self.dn = DecisionNetwork()
        self.dn.build(self.file_info)
        self.d14 = self.dn.vid2nid[14]
        self.d15 = self.dn.vid2nid[15]
        self.d16 = self.dn.vid2nid[16]
        self.d17 = self.dn.vid2nid[17]
        self.u1 = self.dn.fid2nid[14]
        self.u2 = self.dn.fid2nid[15]
        self.u3 = self.dn.fid2nid[16]
        self.u4 = self.dn.fid2nid[17]

    def test_some_statements(self):
        decision_blocks = filter_subsets(rev_topo_sort(self.dn), self.dn.decision_nids)
        self.assertEqual([SortedSet([17]), SortedSet([16]), SortedSet([15]), SortedSet([14])], list(decision_blocks))

    def test_rel_u(self):
        rel_u = relevant_utility(self.dn, rel_d= SortedSet([self.d16]))
        self.assertEqual(SortedSet([self.u2, self.u3, self.u4]), rel_u)

        rel_u = relevant_utility(self.dn, rel_d= SortedSet([self.d17]))
        self.assertEqual(SortedSet([self.u4]), rel_u)

    def test_history(self):
        h1 = self.dn.history_for_decisions( self.d14)
        self.assertEqual(SortedSet([1]), h1)

        h2 = self.dn.history_for_decisions( self.d15)
        self.assertEqual(SortedSet([4, 5, 1, 14]), h2)

        h3 = self.dn.history_for_decisions( self.d16)
        self.assertEqual(SortedSet([1, 4, 5, 14, 15]), h3)

        h4 = self.dn.history_for_decisions( self.d17)
        self.assertEqual(SortedSet([1, self.d14, 4, 5, self.d15, self.d16, 6]), h4)

    def test_backdoor(self):
        res = is_backdoor(self.dn, x=self.d17, y=self.u4, Z=SortedSet([6, self.d15, 4]))
        self.assertTrue(res)

        res = is_backdoor(self.dn, x=self.d17, y=self.u4, Z=SortedSet([6, self.d15]))
        self.assertFalse(res)

        res = is_backdoor(self.dn, x=self.d17, y=self.u4, Z=SortedSet([4, self.d15]))
        self.assertFalse(res)

        res = is_backdoor(self.dn, x=self.d17, y=self.u4, Z=SortedSet([6, 4, self.d16]))
        self.assertFalse(res)

        res = is_backdoor(self.dn, x=self.d17, y=self.u4, Z=SortedSet([6, 4, self.d16, self.d15]))
        self.assertTrue(res)

    def test_backdoor2(self):
        G = DecisionNetwork()
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=0)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=1)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=2)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=3)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=4)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=5)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=6)
        G.add_edges_from([(0,1), (1,2), (1,3), (2,3), (4,1), (4,2), (4,3), (5,1), (5,4), (6,3), (6,4)])
        # backdoors(1,3) = 4,5 4,6
        self.assertTrue(is_backdoor(G, 1, 3, SortedSet([4, 5])) )
        self.assertTrue(is_backdoor(G, 1, 3, SortedSet([4, 6])) )
        self.assertTrue(is_backdoor(G, 1, 3, SortedSet([4, 5, 6])) )
        self.assertTrue(is_backdoor(G, 1, 3, SortedSet([4, 5, 6, 2])) )

        self.assertFalse(is_backdoor(G, 1, 3, SortedSet([5])))
        self.assertFalse(is_backdoor(G, 1, 3, SortedSet([4])))
        self.assertFalse(is_backdoor(G, 1, 3, SortedSet([6])))
        self.assertFalse(is_backdoor(G, 1, 3, SortedSet([0, 4])))
        self.assertFalse(is_backdoor(G, 1, 3, SortedSet([0, 5])))

    def test_frontdoor(self):
        G = DecisionNetwork()
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=0)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=1)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=2)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=3)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=4)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=5)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=6)
        G.add_node_by_nid(type=TYPE_CHANCE_VAR, vid=7)
        G.add_edges_from([(0,1), (1,2), (1,3), (1,6), (2,3), (3,4), (5,1), (5,4), (7,6), (7,2), (7,4)])
        # frontdoor(1,4) = 3,7

        self.assertTrue(is_frontdoor(G, 1, 4, SortedSet([3, 7])))
        self.assertFalse(is_frontdoor(G, 1, 4, SortedSet([2])))

    def test_hidden_nodes(self):
        h1 = [0, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
        self.assertEqual(SortedSet(h1), self.dn.hidden_nodes_for_decisions(self.d16))

    def test_rel_o(self):
        o1 = relevant_observation(self.dn, self.d17, self.u4)
        self.assertEqual(SortedSet([4, 6, self.d15]), o1)

    def test_barren_nodes(self):
        from copy import deepcopy
        dn = deepcopy(self.dn)
        h = dn.hidden_nodes_for_decisions(self.d17)
        u = relevant_utility(dn, self.d17)
        b = barren_nodes(dn, h, u)
        b_ans = [7, 9, 10]
        self.assertEqual(SortedSet(b_ans), b)


    def test_rel_h(self):
        from copy import deepcopy
        dn = deepcopy(self.dn)
        h = dn.hidden_nodes_for_decisions(self.d17)
        u = relevant_utility(dn, self.d17)
        b = barren_nodes(dn, h, u)
        for nid in b:
            dn.remove_node_by_nid(nid)
        h = relevant_hidden(dn, self.d17, u)
        h_ans = [8, 11, 12]
        self.assertEqual(SortedSet(h_ans), h)


    def test_submodel(self):
        from copy import deepcopy
        dn = deepcopy(self.dn)
        u = relevant_utility(dn, self.d17)                  # self.u4
        o = relevant_observation(self.dn, self.d17, u)      # [4, 6, self.d15]
        h = dn.hidden_nodes_for_decisions(self.d17)         # [0, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13]
        b = barren_nodes(dn, h, u)                          # [7, 9, 10]
        for nid in b:
            dn.remove_node_by_nid(nid)
        h = relevant_hidden(dn, self.d17, u)                # [8, 11, 12]
        s = SubModel(self.d17, u, o, h)

        projected_dn = s.relevant_decision_network(dn)
        self.assertEqual(SortedSet([4, 6, self.d15, self.d17]), s.policy_scope)
        interface = s.rel_o_out_from_dn(dn)
        self.assertEqual(SortedSet([4, 15]), interface)
        self.assertEqual(SortedSet([self.d17, self.d15, 4, 6, 8, 11, 12, self.u4]), s.relevant_nodes)
        self.assertTrue(s.is_atomic)
        self.assertFalse(s.is_composite)

        # projected_dn
        # [4, 6, 8, 11, 12, 15, 17, 21]
        # rel_d = 17
        # rel_o = 4, 6, 15
        # rel_h = 8, 11, 12
        # rel_u = 21

    def test_submodel_tree_dec(self):
        # check each node, submodel and edges, messages
        from copy import deepcopy
        dn = deepcopy(self.dn)
        st = submodel_tree_decomposition(dn)    # go through the code


