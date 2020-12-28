import unittest
import os
from sortedcontainers import SortedSet
import numpy as np
from gmid2.basics.uai_files import read_limid
from gmid2.basics.directed_network import DecisionNetwork
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.basics.factor import *
from gmid2.inference.submodel import submodel_tree_decomposition
from gmid2.inference.st_bte import StBTE

class StBteBuildTest(unittest.TestCase):
    def done_test1(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_prob_to_log()    # conversion is done here!
        gm.convert_util_to_alpha()


# fid2f: SortedDict({0: Factor0(1), 1: Factor1(2), 2: Factor2(0,1,3), 3: Factor3(0,2,4), 4: Factor4(0), 5: Factor5(1,5), 6: Factor6(2,5)})
# vid2type: SortedDict({0: 'D', 1: 'C', 2: 'C', 3: 'C', 4: 'C', 5: 'D'})
#
# factors
# 0 (94121824285408) = {Factor} Factor0(1[2]):=[0.8 0.2]
# 1 (94121824285440) = {Factor} Factor1(2[2]):=[0.3 0.7]
# 2 (94121824285472) = {Factor} Factor2(0[2],1[2],3[2]):=[0.2 0.8 0.4 0.6 0.3 0.7 0.6 0.4]
# 3 (94121824285504) = {Factor} Factor3(0[2],2[2],4[2]):=[0.7 0.3 0.5 0.5 0.4 0.6 0.9 0.1]
# 4 (94121824285536) = {Factor} Factor4(0[2]):=[10. 20.]
# 5 (94121824285568) = {Factor} Factor5(1[2],5[2]):=[10. 20. 30. 40.]
# 6 (94121824285600) = {Factor} Factor6(2[2],5[2]):=[50. 10. 30. 20.]
#
# gm.convert_prob_to_log()    # conversion is done here!
# 0 (94857929106144) = {Factor} Factor0(1[2]):=[-0.22314355 -1.60943791]
# 1 (94857929106176) = {Factor} Factor1(2[2]):=[-1.2039728  -0.35667494]
# 2 (94857929106208) = {Factor} Factor2(0[2],1[2],3[2]):=[-1.60943791 -0.22314355 -0.91629073 -0.51082562 -1.2039728  -0.35667494\n -0.51082562 -0.91629073]
# 3 (94857929106240) = {Factor} Factor3(0[2],2[2],4[2]):=[-0.35667494 -1.2039728  -0.69314718 -0.69314718 -0.91629073 -0.51082562\n -0.10536052 -2.30258509]
# 4 (94857929106272) = {Factor} Factor4(0[2]):=[10. 20.]
# 5 (94857929106304) = {Factor} Factor5(1[2],5[2]):=[10. 20. 30. 40.]
# 6 (94857929106336) = {Factor} Factor6(2[2],5[2]):=[50. 10. 30. 20.]
#
# gm.convert_util_to_alpha()
# 0 (94613800383200) = {Factor} Factor0(1[2]):=[0.8 0.2]
# 1 (94613800383232) = {Factor} Factor1(2[2]):=[0.3 0.7]
# 2 (94613800383264) = {Factor} Factor2(0[2],1[2],3[2]):=[0.2 0.8 0.4 0.6 0.3 0.7 0.6 0.4]
# 3 (94613800383296) = {Factor} Factor3(0[2],2[2],4[2]):=[0.7 0.3 0.5 0.5 0.4 0.6 0.9 0.1]
# 4 (94613800383328) = {Factor} Factor4(0[2]):=[1.e-08 2.e-08]
# 5 (94613800383360) = {Factor} Factor5(1[2],5[2]):=[1.e-08 2.e-08 3.e-08 4.e-08]
# 6 (94613800383392) = {Factor} Factor6(2[2],5[2]):=[5.e-08 1.e-08 3.e-08 2.e-08]

    def done_test2(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_prob_to_log()    # conversion is done here!
        gm.convert_util_to_alpha()
        gm.remove_fid(0)

        print(gm.fid2f)

    def done_test_project(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)

        gm2 = gm.project(project_fids=[0,1,2], copy_fids=[0])
        print(gm2)
# 0 (93870337061600) = {Factor} Factor0(1[2]):=[0.8 0.2]
# 1 (93870337061632) = {Factor} Factor1(2[2]):=[0.3 0.7]
# 2 (93870337061664) = {Factor} Factor2(0[2],1[2],3[2]):=[0.2 0.8 0.4 0.6 0.3 0.7 0.6 0.4]
# 3 (93870337061696) = {Factor} Factor3(0[2],2[2],4[2]):=[0.7 0.3 0.5 0.5 0.4 0.6 0.9 0.1]
# 4 (93870337061728) = {Factor} Factor4(0[2]):=[10. 20.]
# 5 (93870337061760) = {Factor} Factor5(1[2],5[2]):=[10. 20. 30. 40.]
# 6 (93870337061792) = {Factor} Factor6(2[2],5[2]):=[50. 10. 30. 20.]
#
# vid2fids = {SortedDict} SortedDict({0: SortedSet([2, 3, 4]), 1: SortedSet([0, 2, 5]), 2: SortedSet([1, 3, 6]), 3: SortedSet([2]), 4: SortedSet([3]), 5: SortedSet([5, 6])})
#  0 (93870337061600) = {SortedSet} SortedSet([2, 3, 4])
#  1 (93870337061632) = {SortedSet} SortedSet([0, 2, 5])
#  2 (93870337061664) = {SortedSet} SortedSet([1, 3, 6])
#  3 (93870337061696) = {SortedSet} SortedSet([2])
#  4 (93870337061728) = {SortedSet} SortedSet([3])
#  5 (93870337061760) = {SortedSet} SortedSet([5, 6])
#
#
#  fid2f = {SortedDict} SortedDict({1: Factor1(2), 2: Factor2(0,1,3), 7: Factor7(1)})
#  1 (93870337061632) = {Factor} Factor1(2[2]):=[0.3 0.7]
#  2 (93870337061664) = {Factor} Factor2(0[2],1[2],3[2]):=[0.2 0.8 0.4 0.6 0.3 0.7 0.6 0.4]
#  7 (93870337061824) = {Factor} Factor7(1[2]):=[0.8 0.2]
#
#  vid2fids = {SortedDict} SortedDict({0: SortedSet([2]), 1: SortedSet([2, 7]), 2: SortedSet([1]), 3: SortedSet([2])})
#  0 (93870337061600) = {SortedSet} SortedSet([2])
#  1 (93870337061632) = {SortedSet} SortedSet([2, 7])
#  2 (93870337061664) = {SortedSet} SortedSet([1])
#  3 (93870337061696) = {SortedSet} SortedSet([2])


    def done_test_copy_subset(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)

        gm2 = gm.copy_subset(copy_fids=[0])
        print(gm2)

# fid2f = {SortedDict} SortedDict({1: Factor1(2), 2: Factor2(0,1,3), 3: Factor3(0,2,4), 4: Factor4(0), 5: Factor5(1,5), 6: Factor6(2,5), 7: Factor7(1)})
#  1 (94762344739584) = {Factor} Factor1(2[2]):=[0.3 0.7]
#  2 (94762344739616) = {Factor} Factor2(0[2],1[2],3[2]):=[0.2 0.8 0.4 0.6 0.3 0.7 0.6 0.4]
#  3 (94762344739648) = {Factor} Factor3(0[2],2[2],4[2]):=[0.7 0.3 0.5 0.5 0.4 0.6 0.9 0.1]
#  4 (94762344739680) = {Factor} Factor4(0[2]):=[10. 20.]
#  5 (94762344739712) = {Factor} Factor5(1[2],5[2]):=[10. 20. 30. 40.]
#  6 (94762344739744) = {Factor} Factor6(2[2],5[2]):=[50. 10. 30. 20.]
#  7 (94762344739776) = {Factor} Factor7(1[2]):=[0.8 0.2]
#
# vid2fids = {SortedDict} SortedDict({0: SortedSet([2, 3, 4]), 1: SortedSet([2, 5, 7]), 2: SortedSet([1, 3, 6]), 3: SortedSet([2]), 4: SortedSet([3]), 5: SortedSet([5, 6])})
#  0 (94762344739552) = {SortedSet} SortedSet([2, 3, 4])
#  1 (94762344739584) = {SortedSet} SortedSet([2, 5, 7])
#  2 (94762344739616) = {SortedSet} SortedSet([1, 3, 6])
#  3 (94762344739648) = {SortedSet} SortedSet([2])
#  4 (94762344739680) = {SortedSet} SortedSet([3])
#  5 (94762344739712) = {SortedSet} SortedSet([5, 6])

    def done_test_change_var_type(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.change_var_type(5, 'C')
        print(gm)
# vid2var = {SortedDict} SortedDict({0: Variable(vid=0, dim=2, type='D'), 1: Variable(vid=1, dim=2, type='C'), 2: Variable(vid=2, dim=2, type='C'), 3: Variable(vid=3, dim=2, type='C'), 4: Variable(vid=4, dim=2, type='C'), 5: Variable(vid=5, dim=2, type='D')})
#  0 (94527475737312) = {Variable} Variable(vid=0, dim=2, type='D')
#  1 (94527475737344) = {Variable} Variable(vid=1, dim=2, type='C')
#  2 (94527475737376) = {Variable} Variable(vid=2, dim=2, type='C')
#  3 (94527475737408) = {Variable} Variable(vid=3, dim=2, type='C')
#  4 (94527475737440) = {Variable} Variable(vid=4, dim=2, type='C')
#  5 (94527475737472) = {Variable} Variable(vid=5, dim=2, type='D')
#
#  vid2var = {SortedDict} SortedDict({0: Variable(vid=0, dim=2, type='D'), 1: Variable(vid=1, dim=2, type='C'), 2: Variable(vid=2, dim=2, type='C'), 3: Variable(vid=3, dim=2, type='C'), 4: Variable(vid=4, dim=2, type='C'), 5: Variable(vid=5, dim=2, type='C')})
#  0 (94527475737312) = {Variable} Variable(vid=0, dim=2, type='D')
#  1 (94527475737344) = {Variable} Variable(vid=1, dim=2, type='C')
#  2 (94527475737376) = {Variable} Variable(vid=2, dim=2, type='C')
#  3 (94527475737408) = {Variable} Variable(vid=3, dim=2, type='C')
#  4 (94527475737440) = {Variable} Variable(vid=4, dim=2, type='C')
#  5 (94527475737472) = {Variable} Variable(vid=5, dim=2, type='C')
#
# 5 (94240602694528) = {Factor} Factor5(1[2],5[2]):=[10. 20. 30. 40.]
#  fid = {int} 5
#  nvar = {int} 2
#  scope = {SortedSet} SortedSet([Variable(vid=1, dim=2, type='C'), Variable(vid=5, dim=2, type='D')])
#
# 6 (94240602694560) = {Factor} Factor6(2[2],5[2]):=[50. 10. 30. 20.]
#  fid = {int} 6
#  nvar = {int} 2
#  scope = {SortedSet} SortedSet([Variable(vid=2, dim=2, type='C'), Variable(vid=5, dim=2, type='D')])
#
#  5 (94527475737472) = {Factor} Factor5(1[2],5[2]):=[10. 20. 30. 40.]
#  fid = {int} 5
#  nvar = {int} 2
#  scope = {SortedSet} SortedSet([Variable(vid=1, dim=2, type='C'), Variable(vid=5, dim=2, type='C')])
#
#  6 (94527475737504) = {Factor} Factor6(2[2],5[2]):=[50. 10. 30. 20.]
#  fid = {int} 6
#  nvar = {int} 2
#  scope = {SortedSet} SortedSet([Variable(vid=2, dim=2, type='C'), Variable(vid=5, dim=2, type='C')])

    def done_test_scope_vids(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        print( gm.scope_vids )

    def done_test_messages_factors1(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)

        f0 = gm.fid2f[0]
        f1 = gm.fid2f[1]
        f2 = gm.fid2f[2]
        f3 = gm.fid2f[3]
        f4 = gm.fid2f[4]
        f5 = gm.fid2f[5]
        f6 = gm.fid2f[6]

        pr = f0*f1*f2*f3
        ut = f5+f6

        v0 = gm.vid2var[0]      # D
        v1 = gm.vid2var[1]
        v2 = gm.vid2var[2]
        v3 = gm.vid2var[3]
        v4 = gm.vid2var[4]
        v5 = gm.vid2var[5]      # D

        F1 = pr * ut
        F2 = F1.sum_marginal(SortedSet([v1, v2]), inplace=False)
        F3 = F2.max_marginal(SortedSet([v5]), inplace=False)
        F4 = F3.sum_marginal(SortedSet([v3, v4]), inplace=False)
        print(F4)

    def done_test_messages_factors2(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)

        f0 = gm.fid2f[0]
        f1 = gm.fid2f[1]
        f2 = gm.fid2f[2]
        f3 = gm.fid2f[3]
        f4 = gm.fid2f[4]
        f5 = gm.fid2f[5]
        f6 = gm.fid2f[6]

        pr = f0 * f1 * f2 * f3
        ut = f5 + f6

        v0 = gm.vid2var[0]  # D
        v1 = gm.vid2var[1]
        v2 = gm.vid2var[2]
        v3 = gm.vid2var[3]
        v4 = gm.vid2var[4]
        v5 = gm.vid2var[5]  # D

        policy_table = [0.5] * 16
        D = Factor([v0, v3, v4, v5], policy_table, factor_type='S')
        F1 = pr * ut * D
        F2 = F1.sum_marginal(SortedSet([v1, v2]), inplace=False)
        F3 = F2.sum_marginal(SortedSet([v5]), inplace=False)
        F4 = F3.sum_marginal(SortedSet([v3, v4]), inplace=False)
        print(F4)

    def test_messages_factors3(self):
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        file_info = read_limid(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_prob_to_log()
        gm.convert_util_to_alpha()

        f0 = gm.fid2f[0]
        f1 = gm.fid2f[1]
        f2 = gm.fid2f[2]
        f3 = gm.fid2f[3]
        f4 = gm.fid2f[4]
        f5 = gm.fid2f[5]
        f6 = gm.fid2f[6]

        pr = f0 + f1 + f2 + f3
        ut = f5 + f6

        v0 = gm.vid2var[0]  # D
        v1 = gm.vid2var[1]
        v2 = gm.vid2var[2]
        v3 = gm.vid2var[3]
        v4 = gm.vid2var[4]
        v5 = gm.vid2var[5]  # D

        policy_table = [0.5] * 16
        D = Factor([v0, v3, v4, v5], policy_table, factor_type='S')
        D.ilog()
        Q = Factor([v0], [0.5, 0.5], factor_type='C')
        Q.ilog()

        F1 = pr + ut + D + Q
        F2 = F1.lse_marginal(SortedSet([v1, v2]), inplace=False)
        F3 = F2.lse_marginal(SortedSet([v5]), inplace=False)
        F4 = F3.lse_marginal(SortedSet([v3, v4]), inplace=False)
        print(F4)

        T1 = f0 + f2 + f5
        T2 = f1 + f3 + f6
        T3 = D + Q

        print("compare with BW cluster factors")