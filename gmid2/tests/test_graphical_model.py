import unittest
import os
from gmid2.basics.uai_files import read_limid, FileInfo
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.basics.factor import Factor, Variable
from sortedcontainers import SortedSet, SortedDict, SortedList
import numpy.testing as nptest

class GmCreateTest(unittest.TestCase):
    def setUp(self):
        self.file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        self.file_info = read_limid(self.file_name, skip_table=False)
        # self.file_info.chance_vars = [1, 2, 3, 4]
        # self.file_info.decision_vars = [0, 5]
        # self.file_info.prob_funcs = [0, 1, 2, 3]
        # self.file_info.util_funcs = [4, 5, 6]
        # self.file_info.tables = [
        #     [0.8, 0.2],
        #     [0.3, 0.7],
        #     [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.6, 0.4],
        #     [0.7, 0.3, 0.4, 0.6, 0.5, 0.5, 0.9, 0.1],
        #     [10.0, 20.0],
        #     [10.0, 20.0, 30.0, 40.0],
        #     [50.0, 10.0, 30.0, 20.0]
        # ]
        # self.file_info.scopes = [[1], [2], [1, 0, 3], [2, 0, 4], [0], [1, 5], [2, 5]]
        # self.file_info.factor_types = ['P', 'P', 'P', 'P', 'U', 'U', 'U']


    def assert_equal_factor(self, f1, f2):
        self.assertEqual(f1.scope, f2.scope)
        self.assertEqual(f1.scope_vars, f2.scope_vars)
        nptest.assert_almost_equal(f1.table, f2.table)
        self.assertEqual(f1.type, f2.type)


    def test_build_fid2_f(self):
        print(self.id())
        Factor.reset_fid()
        gm = GraphicalModel()
        gm.build(self.file_info)
        f_dict = SortedDict()
        f_dict[0] = Factor([Variable(1, 2, 'C')], [0.8, 0.2], 'P')
        f_dict[1] = Factor([Variable(2, 2, 'C')], [0.3, 0.7], 'P')
        f_dict[2] = Factor([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(3, 2, 'C')],
                           [0.2, 0.8, 0.4, 0.6, 0.3, 0.7, 0.6, 0.4], 'P')
        f_dict[3] = Factor([Variable(0, 2, 'D'), Variable(2, 2, 'C'), Variable(4, 2, 'C')],
                           [0.7, 0.3, 0.5, 0.5, 0.4, 0.6, 0.9, 0.1], 'P')
        f_dict[4] = Factor([Variable(0, 2, 'D')], [10.0, 20.0], 'U')
        f_dict[5] = Factor([Variable(1, 2, 'C'), Variable(5, 2, 'D')], [10.0, 20.0, 30.0, 40.0], 'U')
        f_dict[6] = Factor([Variable(2, 2, 'C'), Variable(5, 2, 'D')], [50.0, 10.0, 30.0, 20.0], 'U')

        for i in range(7):
            f1 = gm.fid2f[i]
            f2 = f_dict[i]
            print("\t{} == {}".format(f1, f2))
            self.assert_equal_factor(f1, f2)

        # self.assertEqual(gm.util_fids, SortedSet([4, 5, 6]))
        # self.assertEqual(gm.prob_fids, SortedSet([0, 1, 2, 3]))
        # self.assertEqual(gm.policy_fids, SortedSet())


    def test_build_vid2_fs(self):
        print(self.id())
        Factor.reset_fid()
        gm = GraphicalModel()
        gm.build(self.file_info)

        self.assertEqual(gm.vid2fids[0], SortedSet([2, 3, 4]))
        self.assertEqual(gm.vid2fids[1], SortedSet([0, 2, 5]))
        self.assertEqual(gm.vid2fids[2], SortedSet([1, 3, 6]))
        self.assertEqual(gm.vid2fids[3], SortedSet([2]))
        self.assertEqual(gm.vid2fids[4], SortedSet([3]))
        self.assertEqual(gm.vid2fids[5], SortedSet([5, 6]))

        # self.assertEqual(gm.vid2_dim, SortedList([2]*6))
        # self.assertEqual(gm.vid2_type, SortedList(['D', 'C', 'C', 'C', 'C', 'D']))
        # self.assertEqual(gm.vid2_weight, SortedList([0.0, 1.0, 1.0, 1.0, 1.0, 0.0]))

        # self.assertEqual(gm.chance_vids, SortedSet([1, 2, 3, 4]))
        # self.assertEqual(gm.decision_vids, SortedSet([0, 5]))

    def test_getitem(self):
        print(self.id())
        Factor.reset_fid()
        gm = GraphicalModel()
        gm.build(self.file_info)
        f_dict = SortedDict()
        f_dict[0] = Factor([Variable(1, 2, 'C')], [0.8, 0.2], 'P')
        f_dict[1] = Factor([Variable(2, 2, 'C')], [0.3, 0.7], 'P')
        f_dict[2] = Factor([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(3, 2, 'C')],
                           [0.2, 0.8, 0.4, 0.6, 0.3, 0.7, 0.6, 0.4], 'P')
        f_dict[3] = Factor([Variable(0, 2, 'D'), Variable(2, 2, 'C'), Variable(4, 2, 'C')],
                           [0.7, 0.3, 0.5, 0.5, 0.4, 0.6, 0.9, 0.1], 'P')
        f_dict[4] = Factor([Variable(0, 2, 'D')], [10.0, 20.0], 'U')
        f_dict[5] = Factor([Variable(1, 2, 'C'), Variable(5, 2, 'D')], [10.0, 20.0, 30.0, 40.0], 'U')
        f_dict[6] = Factor([Variable(2, 2, 'C'), Variable(5, 2, 'D')], [50.0, 10.0, 30.0, 20.0], 'U')

        for fid in range(7):
            self.assert_equal_factor(gm[fid], f_dict[fid])

        temp = gm[0, 1, 2]
        self.assert_equal_factor(gm[0], temp[0])
        self.assert_equal_factor(gm[1], temp[1])
        self.assert_equal_factor(gm[2], temp[2])

        temp = gm[0, 2, 5, 6]
        self.assert_equal_factor(gm[0], temp[0])
        self.assert_equal_factor(gm[2], temp[1])
        self.assert_equal_factor(gm[5], temp[2])
        self.assert_equal_factor(gm[6], temp[3])

        temp = gm[4, 3, 2, 1]
        self.assert_equal_factor(gm[4], temp[0])
        self.assert_equal_factor(gm[3], temp[1])
        self.assert_equal_factor(gm[2], temp[2])
        self.assert_equal_factor(gm[1], temp[3])

        temp = gm[(4, 3, 2, 1)]
        self.assert_equal_factor(gm[4], temp[0])
        self.assert_equal_factor(gm[3], temp[1])
        self.assert_equal_factor(gm[2], temp[2])
        self.assert_equal_factor(gm[1], temp[3])

        temp = gm[ [4, 3, 2, 1] ]       # still works in python
        self.assert_equal_factor(gm[4], temp[0])
        self.assert_equal_factor(gm[3], temp[1])
        self.assert_equal_factor(gm[2], temp[2])
        self.assert_equal_factor(gm[1], temp[3])

        temp = gm[ range(4)]            # still works in python
        self.assert_equal_factor(gm[0], temp[0])
        self.assert_equal_factor(gm[1], temp[1])
        self.assert_equal_factor(gm[2], temp[2])
        self.assert_equal_factor(gm[3], temp[3])


class GmFactorInterfaceTest(unittest.TestCase):
    pass


class GmVariableInterfaceTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
