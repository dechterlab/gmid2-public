import unittest
import os
from sortedcontainers import SortedSet, SortedDict, SortedList
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.basics.uai_files import read_limid, FileInfo
from gmid2.basics.directed_network import DecisionNetwork
from gmid2.global_constants import *


class DnCreationTest(unittest.TestCase):
    def setUp(self):
        self.file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        self.file_info = read_limid(self.file_name, skip_table=False)

    def test_creation(self):
        print("\n\n{}".format(self.id()))
        dn = DecisionNetwork()
        dn.build(self.file_info)

        self.assertEqual(dn.name, "car.uai")
        self.assertEqual(dn.net_type, TYPE_ID_NETWORK)
        # self.assertEqual(dn.nblock, 4)
        t = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0, 5: 5, 6: None, 7: None, 8: None}
        temp = SortedDict({k: dn.nid2vid(k) for k in t})
        self.assertEqual(temp, SortedDict(t))
        t = {0: 0, 1: 1, 2: 2, 3: 3, 4:None, 5: None, 6: 4, 7: 5, 8: 6}
        temp = SortedDict({k: dn.nid2fid(k) for k in t})
        self.assertEqual(temp, SortedDict(t))

        # pp.pprint(dict(dn.nodes))
        # _node dict
        # 0 (94086376674016) = {dict} {'vid': 1, 'vid_t': 'C', 'fid': 0, 'fid_t': 'P', 'conditioned': False, 'block': 0}
        # 1 (94086376674048) = {dict} {'vid': 2, 'vid_t': 'C', 'fid': 1, 'fid_t': 'P', 'conditioned': False, 'block': 0}
        # 2 (94086376674080) = {dict} {'vid': 3, 'vid_t': 'C', 'fid': 2, 'fid_t': 'P', 'conditioned': False, 'block': 2}
        # 3 (94086376674112) = {dict} {'vid': 4, 'vid_t': 'C', 'fid': 3, 'fid_t': 'P', 'conditioned': False, 'block': 2}
        # 4 (94086376674144) = {dict} {'vid': 0, 'vid_t': 'D', 'fid': None, 'fid_t': 'S', 'conditioned': False, 'block': 3}
        # 5 (94086376674176) = {dict} {'vid': 5, 'vid_t': 'D', 'fid': None, 'fid_t': 'S', 'conditioned': False, 'block': 1}
        # 6 (94086376674208) = {dict} {'vid': None, 'vid_t': None, 'fid': 4, 'fid_t': 'U', 'conditioned': False}
        # 7 (94086376674240) = {dict} {'vid': None, 'vid_t': None, 'fid': 5, 'fid_t': 'U', 'conditioned': False}
        # 8 (94086376674272) = {dict} {'vid': None, 'vid_t': None, 'fid': 6, 'fid_t': 'U', 'conditioned': False}

        # pp.pprint(dict(dn.adj))
        # _adj = {dict}
        # 0(94236569617120) = {dict}
        # {2: {'edge_t': 'P'}, 7: {'edge_t': 'U'}}
        # 1(94236569617152) = {dict}
        # {3: {'edge_t': 'P'}, 8: {'edge_t': 'U'}}
        # 2(94236569617184) = {dict}
        # {5: {'edge_t': 'I'}}
        # 3(94236569617216) = {dict}
        # {5: {'edge_t': 'I'}}
        # 4(94236569617248) = {dict}
        # {2: {'edge_t': 'P'}, 3: {'edge_t': 'P'}, 6: {'edge_t': 'U'}}
        # 5(94236569617280) = {dict}
        # {7: {'edge_t': 'U'}, 8: {'edge_t': 'U'}}
        # 6(94236569617312) = {dict}
        # {}
        # 7(94236569617344) = {dict}
        # {}
        # 8(94236569617376) = {dict}
        # {}

    def test_add_node_by_id(self):
        print("\n\n{}".format(self.id()))
        dn0 = DecisionNetwork()
        dn0.build(self.file_info)

        dn = DecisionNetwork()
        dn.add_node_by_nid(type=TYPE_CHANCE_NODE, vid=1, fid=0)     #0
        dn.add_node_by_nid(type=TYPE_CHANCE_NODE, vid=2, fid=1)     #1
        dn.add_node_by_nid(type=TYPE_CHANCE_NODE, vid=3, fid=2)     #2
        dn.add_node_by_nid(type=TYPE_CHANCE_NODE, vid=4, fid=3)     #3
        dn.add_node_by_nid(type=TYPE_DECISION_NODE, vid=0)          #4 d
        dn.add_node_by_nid(type=TYPE_DECISION_NODE, vid=5)          #5 d
        dn.add_node_by_nid(type=TYPE_VALUE_NODE, fid=4)             #6 u
        dn.add_node_by_nid(type=TYPE_VALUE_NODE, fid=5)             #7 u
        dn.add_node_by_nid(type=TYPE_VALUE_NODE, fid=6)             #8 u

        dn.add_edges_by_nid(0, parent_nids=[], child_nids=[2, 7])
        dn.add_edges_by_nid(1, parent_nids=[], child_nids=[3, 8])
        dn.add_edges_by_nid(2, parent_nids=[0, 4], child_nids=[5])
        dn.add_edges_by_nid(3, parent_nids=[1, 4], child_nids=[5])
        dn.add_edges_by_nid(4, parent_nids=[], child_nids=[2, 3, 6])
        dn.add_edges_by_nid(5, parent_nids=[2, 3, 4], child_nids=[7, 8])
        dn.add_edges_by_nid(6, parent_nids=[4], child_nids=[])
        dn.add_edges_by_nid(7, parent_nids=[0, 5], child_nids=[])
        dn.add_edges_by_nid(8, parent_nids=[1, 5], child_nids=[])
        #
        self.assertDictEqual(dict(dn.nodes), dict(dn0.nodes))
        self.assertDictEqual(dict(dn.adj), dict(dn0.adj))

    def test_remove_node_by_id(self):
        print("\n\n{}".format(self.id()))
        dn = DecisionNetwork()
        dn.build(self.file_info)

        dn.remove_node_by_vid(1)
        # pp.pprint(dict(dn.nodes))       # vid1 is n0
        # pp.pprint(dict(dn.adj))         # removing no removes arc to (0, 2) and (0, 7)

        dn.remove_node_by_fid(5)
        # pp.pprint(dict(dn.nodes))       # fid5 is n7, utility
        # pp.pprint(dict(dn.adj))         # removing no removes arc to (5, 7) and (0, 7)


    def test_node_condition(self):
        print("\n\n{}".format(self.id()))
        dn = DecisionNetwork()
        dn.build(self.file_info)

        dn.condition_nodes(0)           # nid0 is vid1
        self.assertTrue(dn.nodes[0]['conditioned'])
        dn.condition_nodes([1,2,3])
        self.assertTrue(dn.nodes[1]['conditioned'])
        self.assertTrue(dn.nodes[2]['conditioned'])
        self.assertTrue(dn.nodes[3]['conditioned'])
        # pp.pprint(dict(dn.nodes))

        dn.uncondition_nodes(0)
        self.assertFalse(dn.nodes[0]['conditioned'])
        dn.uncondition_nodes([1,2])
        self.assertFalse(dn.nodes[1]['conditioned'])
        self.assertFalse(dn.nodes[2]['conditioned'])

    def test_hidden_nodes(self):
        print("\n\n{}".format(self.id()))
        dn = DecisionNetwork()
        dn.build(self.file_info)
        h1 = dn.hidden_nodes_for_decisions([5])
        self.assertEqual(h1, SortedSet([0, 1]))
