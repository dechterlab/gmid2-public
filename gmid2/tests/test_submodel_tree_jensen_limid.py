import unittest
import os
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)


from gmid2.global_constants import *
from gmid2.basics.uai_files import read_limid
from gmid2.basics.helper import filter_subsets
from gmid2.basics.directed_network import *
from gmid2.inference.submodel import *


class SubmodelTreeTestLIMID(unittest.TestCase):
    def setUp(self):
        self.file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "jensen")
        self.file_info = read_limid(self.file_name, skip_table=False)
        self.dn = DecisionNetwork()
        self.dn.build(self.file_info)

    def test_submodel_tree_dec(self):
        st = submodel_tree_decomposition(self.dn)    # go through the code todo decomposition loop bug for multi-dec

        print(st)


