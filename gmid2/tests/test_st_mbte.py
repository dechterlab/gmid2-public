import unittest
import os
from gmid2.basics.uai_files import read_limid
from gmid2.basics.directed_network import DecisionNetwork
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.submodel import submodel_tree_decomposition
from gmid2.inference.st_mbte import StMBTE

class StBteBuildTest(unittest.TestCase):
    def test1(self):
        # read file and do submodel tree decomposition
        # file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "ID_from_BN_78_w23d6")
        # file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "mdp1-4_2_2_5")
        # file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "pomdp1-4_2_2_2_3")
        TEST_PATH = os.path.join(os.getcwd(), "test_data")
        file_name = os.path.join(TEST_PATH, "car" +"_round_norm_10")
        file_info = read_limid(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_prob_to_log()
        gm.convert_util_to_alpha()

        dn = DecisionNetwork()
        dn.build(file_info)
        st = submodel_tree_decomposition(dn)
        print(dn.value_nids)
        # create StBTE message passing object
        st_bte = StMBTE(gm, st, 1)          # __init__ for message passing and nx.DiGraph
        st_bte.build_message_graph()
        st_bte.schedule()
        st_bte.init_propagate()
        st_bte.propagate_iter()
        # bound = st_bte.bounds()