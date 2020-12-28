import unittest
import os
from gmid2.global_constants import *
from gmid2.basics.uai_files import read_limid
from gmid2.basics.directed_network import DecisionNetwork
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.submodel import submodel_tree_decomposition
from gmid2.inference.st_bte_bw import StBTEBw

class StBteBuildTest(unittest.TestCase):
    def test1(self):
        # TEST_PATH = os.path.join(BENCHMARK_DIR, "synthetic")
        # file_name = os.path.join(TEST_PATH, "mdp1-4_2_2_5")
        TEST_PATH = os.path.join(os.getcwd(), "test_data")
        file_name = os.path.join(TEST_PATH, "ID_from_BN_78_w18d3_round_norm_10")

        file_info = read_limid(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_prob_to_log()
        gm.convert_util_to_alpha(1.0)

        dn = DecisionNetwork()
        dn.build(file_info)
        st = submodel_tree_decomposition(dn)
        print(dn.value_nids)

        st_mp = StBTEBw(gm, st, alpha=1.0)          # __init__ for message passing and nx.DiGraph
        st_mp.build_message_graph()
        st_mp.schedule()
        st_mp.init_propagate()
        st_mp.propagate_iter()
        bound = st_mp.bounds()
        print(bound)