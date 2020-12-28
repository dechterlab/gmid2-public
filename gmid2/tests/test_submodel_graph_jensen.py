import unittest
import os
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.global_constants import *
from gmid2.basics.uai_files import read_limid
from gmid2.basics.helper import filter_subsets
from gmid2.basics.directed_network import *
from gmid2.inference.submodel import *


# submodel graph decomposition reads a submodel tree, like join graph decomposition reads a mini-bucket tree
# it modifies the submodel tree to submodel graph by splitting individual submodels
# from the last stage submodel following the reverse topological order of the submodel tree
#   for each submodel id, access submodel object
#       partition the value functions in the submodel
#
#       transfrom input submodel tree to something like mini-cluster tree
#       each node (submodel) directs submodel graph (graph of submodels) using self.sg_
#       self.dn_ still reference the decision network object copied
#
#       for each partition of value nodes, use (rel_d and partition_u) to define submodel
#           mini_submodel only defined over rel_d and partition_u but it is stay inside submodel.dn_
#           call submodel.refine_from_rel_u(dn)
#           this refinement recompute rel_o, rel_h
#           then, relevent_decision_network() can be called on global or local decision network
#           add a mini-submodel per partition to submodel graph
#
#       this partitioning applys to 1 submodel in a submodel tree
#       from a super submodel, we know the scope of the policy by submodel.policy_scope
#       for each mini submodel, mini_submodel.policy_scope may be smaller
#       Here, what we will enforce is the consistency up to expectation


class SubmodelGraphTestID(unittest.TestCase):
    def setUp(self):
        self.file_name = os.path.join(os.path.join(os.getcwd(), "test_data/norm"), "jensen2_norm_0.1")
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
        self.u5 = self.dn.fid2nid[18]
        self.u6 = self.dn.fid2nid[19]
        self.u7 = self.dn.fid2nid[20]
        self.u8 = self.dn.fid2nid[21]
        self.st = submodel_tree_decomposition(self.dn)  # go through the code

    def test_submodel_graph_decomposition(self):
        sg = submodel_graph_decomposition(self.st, i_bound=2, m_bound=3)

