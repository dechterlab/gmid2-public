import os
from gmid2.basics.uai_files import read_limid
from gmid2.basics.directed_network import DecisionNetwork
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.basics.factor import Factor
from gmid2.inference.submodel import submodel_tree_decomposition
from gmid2.inference.st_bte import StBTE

TEST_PATH = os.path.join(os.getcwd(),"test_data")
for f in os.listdir(TEST_PATH):
    if f.endswith("_round_norm_10.uai"):
        print("STRAT {}".format(f))
        file_name = os.path.join(TEST_PATH, f.replace(".uai", ""))
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
        st_bte = StBTE(gm, st)          # __init__ for message passing and nx.DiGraph
        st_bte.build_message_graph()
        st_bte.schedule()
        st_bte.init_propagate()
        st_bte.propagate_iter()
        bound = st_bte.bounds()
        print("{}\t\t{}".format(f, bound))
