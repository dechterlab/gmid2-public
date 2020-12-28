import unittest
import os
from gmid2.basics.uai_files import read_limid
from gmid2.basics.directed_network import DecisionNetwork
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.submodel import submodel_tree_decomposition
from gmid2.inference.st_bte_alpha import StBTEAlpha

class StBteBuildTest(unittest.TestCase):
    def test1(self):
        # read file and do submodel tree decomposition
        # file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "ID_from_BN_78_w23d6")
        # file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "mdp1-4_2_2_5")
        # file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "pomdp1-4_2_2_2_3")
        TEST_PATH = os.path.join(os.getcwd(), "test_data")
        file_name = os.path.join(TEST_PATH, "jensen_round_norm_10")
        file_info = read_limid(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_prob_to_log()    # conversion is done here!
        gm.convert_util_to_alpha()


        dn = DecisionNetwork()
        dn.build(file_info)
        st = submodel_tree_decomposition(dn)
        print(dn.value_nids)
        # create StBTE message passing object
        st_bte = StBTEAlpha(gm, st)          # __init__ for message passing and nx.DiGraph
        st_bte.build_message_graph()
        st_bte.schedule()
        st_bte.init_propagate()
        st_bte.propagate_iter()
        # bound = st_bte.bounds()


# number of connected components in primal graph:1
# MBE
# START
# name:jensen
# num vars:18
# num factors:18
# max domain:2
# max scope:3
# induced width:5
# connected components:1
# final Z:1.0
# final MEU:53.354624
# Finish JOB:jensen, with i-bound 10 at 2020-05-12 15:09:46.949285
# total time (sec) for solving problem:0.0396890640259
# after dividing by 40
# final Z:1.0
# final MEU:1.3338656

# Car problem after normalization
# final Z:1.0
# final MEU:1.4


# from gmid2
# jensen ALPHA 1e-9
# 1.3338657955941358        diff 1e-7


###############
# START
# name:mdp1-4_2_2_5
# num vars:25
# num factors:30
# max domain:2
# max scope:4
# induced width:5
# connected components:1
# final Z:1.0
# final MEU:3.6950570193
