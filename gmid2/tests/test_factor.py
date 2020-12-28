import unittest
import numpy as np
import numpy.testing as nptest
import os
from gmid2.basics.uai_files import read_limid
from gmid2.basics.factor import *
from sortedcontainers import SortedSet


class FactorTest(unittest.TestCase):
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

    def test_create_factor(self):
        print(self.id())
        n = 0
        table = self.file_info.tables[n]
        scope = self.file_info.scopes[n]
        domains = self.file_info.domains
        scope_var = [Variable(vid=v, dim=domains[v], type=self.file_info.var_types[v]) for v in scope ]
        type = self.file_info.factor_types[n]
        f1 = Factor(scope_var, table, type)
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        self.assertEqual(len(f1), 2)
        nptest.assert_almost_equal( np.array([0.8, 0.2]), f1.table )
        self.assertEqual(f1.nvar, 1)
        self.assertEqual(f1.size, 2)
        self.assertEqual(f1.scope_vars, SortedSet([Variable(1,2, 'C')]))
        self.assertEqual(f1.shape, (2,))
        self.assertEqual(SortedSet([Variable(1,2, 'C')]), f1.scope)
        self.assertEqual(f1.type, 'P')
        n = 2
        table = self.file_info.tables[n]
        scope = self.file_info.scopes[n]
        domains = self.file_info.domains
        scope_var = [Variable(vid=v, dim=domains[v], type=self.file_info.var_types[v]) for v in scope ]
        type = self.file_info.factor_types[n]
        f2 = Factor(scope_var, table, type)
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        self.assertEqual(len(f2), 8)
        temp = np.array( [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.6, 0.4])
        temp = temp.reshape(2,2,2)
        temp = temp.transpose( (1,0,2) )
        nptest.assert_almost_equal( temp, f2.table )
        self.assertEqual(f2.nvar, 3)
        self.assertEqual(f2.size, 8)
        self.assertEqual(f2.scope_vars, SortedSet([Variable(0,2, 'D'), Variable(1,2, 'C'), Variable(3,2, 'C')]))
        self.assertEqual(f2.shape, (2, 2, 2))
        self.assertEqual(SortedSet([Variable(0,2, 'D'), Variable(1,2, 'C'), Variable(3,2, 'C')]), f2.scope)
        self.assertEqual(f2.type, 'P')
        n = 6
        table = self.file_info.tables[n]
        scope = self.file_info.scopes[n]
        domains = self.file_info.domains
        scope_var = [Variable(vid=v, dim=domains[v], type=self.file_info.var_types[v]) for v in scope ]
        type = self.file_info.factor_types[n]
        f3 = Factor(scope_var, table, type)
        print("\tstr={}, repr={}".format(str(f3), repr(f3)))
        self.assertEqual(len(f3), 4)
        temp = np.array( [50.0, 10.0, 30.0, 20.0] )
        temp = temp.reshape(2,2)
        temp = temp.transpose( (0,1) )
        nptest.assert_almost_equal( temp, f3.table )
        self.assertEqual(f3.nvar, 2)
        self.assertEqual(f3.size, 4)
        self.assertEqual(f3.scope_vars, SortedSet([Variable(2,2, 'C'), Variable(5,2, 'D')]))
        self.assertEqual(f3.shape, (2, 2))
        self.assertEqual(SortedSet([Variable(2,2, 'C'), Variable(5,2, 'D')]), f3.scope)
        self.assertEqual(f3.type, 'U')

    def test_create_scalar_factor(self):
        print(self.id())
        f = Factor([], 10, 'U')
        print("\tstr={}, repr={}".format(str(f), repr(f)))
        nptest.assert_almost_equal(np.array([10.0]), f.table)
        self.assertEqual(len(f), 1)
        self.assertEqual(f.shape, ())       # scalar factor doesn't have shape () having shape means there is a variable
        self.assertEqual(f.type, 'U')

    def test_unary_magic_abs(self):
        print(self.id())
        f = Factor( [Variable(2,3, 'C'), Variable(0,2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'P')
        print("\tstr={}, repr={}".format(str(f), repr(f)))
        temp = np.array( [0.1, 0.3, -0.5, 0.2, -0.4, -0.6]).reshape((2,3))
        nptest.assert_almost_equal(temp, f.table)
        f2 = abs(f)
        temp = np.array([0.1, 0.3, 0.5, 0.2, 0.4, 0.6]).reshape((2, 3))
        nptest.assert_almost_equal(temp, f2.table)
        self.assertNotEqual(f.fid, f2.fid)
        self.assertEqual(f.type, 'P')
        self.assertEqual(f.scope, SortedSet([Variable(0,2, 'D'), Variable(2,3, 'C')]) )
        self.assertEqual(f.scope, f2.scope)
        self.assertEqual(f.type, f2.type)

    def test_unary_magic_neg(self):
        print(self.id())
        f = Factor( [Variable(2,3, 'C'), Variable(0,2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'P')
        print("\tstr={}, repr={}".format(str(f), repr(f)))
        temp = np.array( [0.1, 0.3, -0.5, 0.2, -0.4, -0.6]).reshape((2,3))
        nptest.assert_almost_equal(temp, f.table)
        f2 = -f
        temp = np.array([-0.1, -0.3, 0.5, -0.2, 0.4, 0.6]).reshape((2, 3))
        nptest.assert_almost_equal(temp, f2.table)
        self.assertNotEqual(f.fid, f2.fid)
        self.assertEqual(f.type, 'P')
        self.assertEqual(f.scope, SortedSet([Variable(0,2, 'D'), Variable(2,3, 'C')]) )
        self.assertEqual(f.scope, f2.scope)
        self.assertEqual(f.type, f2.type)

    def test_unary_magic_pow(self):
        print(self.id())
        f = Factor( [Variable(2,3, 'C'), Variable(0,2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'P')
        print("\tstr={}, repr={}".format(str(f), repr(f)))
        temp = np.array( [0.1, 0.3, -0.5, 0.2, -0.4, -0.6]).reshape((2,3))
        nptest.assert_almost_equal(temp, f.table)
        f2 = f**2
        temp = np.array([0.01, 0.09, 0.25, 0.04, 0.16, 0.36]).reshape((2, 3))
        nptest.assert_almost_equal(temp, f2.table)
        self.assertNotEqual(f.fid, f2.fid)
        self.assertEqual(f.type, 'P')
        self.assertEqual(f.scope, SortedSet([Variable(0,2, 'D'), Variable(2,3, 'C')]) )
        self.assertEqual(f.scope, f2.scope)
        self.assertEqual(f.type, f2.type)

    def test_unary_exp(self):
        print(self.id())
        f = Factor( [Variable(2,3, 'C'), Variable(0,2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'P')
        print("\tstr={}, repr={}".format(str(f), repr(f)))
        temp = np.array( [0.1, 0.3, -0.5, 0.2, -0.4, -0.6]).reshape((2,3))
        nptest.assert_almost_equal(temp, f.table)
        f2 = f.exp()
        temp = np.exp(np.array([0.1, 0.3, -0.5, 0.2, -0.4, -0.6])).reshape((2, 3))
        nptest.assert_almost_equal(temp, f2.table)
        self.assertNotEqual(f.fid, f2.fid)
        self.assertEqual(f.type, 'P')
        self.assertEqual(f.scope, SortedSet([Variable(0,2, 'D'), Variable(2,3, 'C')]) )
        self.assertEqual(f.scope, f2.scope)
        self.assertEqual(f.type, f2.type)

    def test_unary_iexp(self):
        print(self.id())
        f = Factor( [Variable(2,3, 'C'), Variable(0,2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'P')
        print("\tstr={}, repr={}".format(str(f), repr(f)))
        temp = np.array( [0.1, 0.3, -0.5, 0.2, -0.4, -0.6]).reshape((2,3))
        nptest.assert_almost_equal(temp, f.table)
        f.iexp()
        temp = np.exp(np.array([0.1, 0.3, -0.5, 0.2, -0.4, -0.6])).reshape((2, 3))
        nptest.assert_almost_equal(temp, f.table)
        self.assertEqual(f.type, 'P')
        self.assertEqual(f.scope, SortedSet([Variable(0,2, 'D'), Variable(2,3, 'C')]) )

    def test_unary_iabs(self):
        print(self.id())
        f = Factor( [Variable(2,3, 'C'), Variable(0,2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'P')
        print("\tstr={}, repr={}".format(str(f), repr(f)))
        temp = np.array( [0.1, 0.3, -0.5, 0.2, -0.4, -0.6]).reshape((2,3))
        nptest.assert_almost_equal(temp, f.table)
        f.iabs()
        temp = np.array([0.1, 0.3, 0.5, 0.2, 0.4, 0.6]).reshape((2, 3))
        nptest.assert_almost_equal(temp, f.table)
        self.assertEqual(f.type, 'P')
        self.assertEqual(f.scope, SortedSet([Variable(0,2, 'D'), Variable(2,3, 'C')]) )

    def test_unary_ipow(self):
        print(self.id())
        f = Factor( [Variable(2,3, 'C'), Variable(0,2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'P')
        print("\tstr={}, repr={}".format(str(f), repr(f)))
        temp = np.array( [0.1, 0.3, -0.5, 0.2, -0.4, -0.6]).reshape((2,3))
        nptest.assert_almost_equal(temp, f.table)
        f.ipow(2.0)
        temp = np.power(np.array([0.1, 0.3, -0.5, 0.2, -0.4, -0.6]), 2.0).reshape((2, 3))
        nptest.assert_almost_equal(temp, f.table)
        self.assertEqual(f.type, 'P')
        self.assertEqual(f.scope, SortedSet([Variable(0,2, 'D'), Variable(2,3, 'C')]) )

    def test_contains(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C'), Variable(1, 2, 'C')], [-1.0, -2.0, -3.0, 4.0, 5.0, 6.0], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        f3 = f1 + f2
        print("\tstr={}, repr={}".format(str(f3), repr(f3)))

        self.assertTrue(0 in f3)
        self.assertTrue(1 in f3)
        self.assertTrue(2 in f3)
        self.assertTrue(3 not in f3)

        self.assertTrue(Variable(0, 2, 'D') in f3)
        self.assertTrue(Variable(1, 2, 'C') in f3)
        self.assertTrue(Variable(2, 3, 'C') in f3)
        self.assertTrue(Variable(2, 4, 'C') not in f3)
        self.assertTrue(Variable(3, 2, 'C') not in f3)

    def test_getsetitem(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C'), Variable(1, 2, 'C')], [-1.0, -2.0, -3.0, 4.0, 5.0, 6.0], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        f3 = f1 + f2
        print("\tstr={}, repr={}".format(str(f3), repr(f3)))
        self.assertAlmostEqual(f3[0,0,0], -0.9)
        self.assertAlmostEqual(f3[1,0,0], -0.8)


        f3[0,0,0] = 100.0
        f3[1,0,0] = 20.0
        t4 = np.array(
            [
                [
                    [100.0, -2.7, 4.5],
                    [-1.9, 4.3, 5.5]
                ],

                [
                    [20.0, -3.4, 4.4],
                    [-1.8, 3.6, 5.4]
                ]
            ]
        )
        nptest.assert_almost_equal(t4, f3.table)

        f3[0] = 10.0
        t5 = np.array(
            [
                [
                    [10.0, 10.0, 10.0],
                    [10.0, 10.0, 10.0]
                ],

                [
                    [20.0, -3.4, 4.4],
                    [-1.8, 3.6, 5.4]
                ]
            ]
        )
        nptest.assert_almost_equal(t5, f3.table)

        f3[0,:,1] = 99.0
        t6 = np.array(
            [
                [
                    [10.0, 99.0, 10.0],
                    [10.0, 99.0, 10.0]
                ],

                [
                    [20.0, -3.4, 4.4],
                    [-1.8, 3.6, 5.4]
                ]
            ]
        )
        nptest.assert_almost_equal(t6, f3.table)

    def test_binary_magic_add(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C'), Variable(1, 2, 'C')], [-1.0, -2.0, -3.0, 4.0, 5.0, 6.0], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        f3 = f1 + f2
        print("\tstr={}, repr={}".format(str(f3), repr(f3)))

        scope3 = SortedSet( [Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(2,3, 'C')] )
        self.assertEqual( scope3, f3.scope)

        t1 = np.array( [ [0.1, 0.3, -0.5], [0.2, -0.4, -0.6]] )
        t2 = np.array( [[-1, -3, 5], [-2, 4, 6]] )
        nptest.assert_almost_equal(t1, f1.table)
        nptest.assert_almost_equal(t2, f2.table)

        t3 = np.array(
            [
                [
                    [-0.9, -2.7, 4.5],
                    [-1.9, 4.3, 5.5]
                ],

                [
                    [-0.8, -3.4, 4.4],
                    [-1.8, 3.6, 5.4]
                ]
            ]
        )
        nptest.assert_almost_equal(t3, f3.table)

    def test_binary_magic_radd(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = 1 + f1
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        self.assertEqual(f1.scope, f2.scope)
        self.assertNotEqual(f1.fid, f2.fid)

    def test_binary_magic_add_const(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = f1 + 1
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        self.assertEqual(f1.scope, f2.scope)
        self.assertNotEqual(f1.fid, f2.fid)

    def test_binary_magic_iadd_const(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f1 += 1
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))

    def test_binary_magic_iadd_expand(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C'), Variable(1, 2, 'C')], [-1.0, -2.0, -3.0, 4.0, 5.0, 6.0], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))

        f1 += f2
        scope = SortedSet([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(2, 3, 'C')])
        self.assertEqual(scope, f1.scope)
        t3 = np.array(
            [
                [
                    [-0.9, -2.7, 4.5],
                    [-1.9, 4.3, 5.5]
                ],

                [
                    [-0.8, -3.4, 4.4],
                    [-1.8, 3.6, 5.4]
                ]
            ]
        )
        nptest.assert_almost_equal(t3, f1.table)
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))

    def test_binary_magic_iadd_sub(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [1, 2, 3, 4, 5, 6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C')], [10.0, 20.0, 30.0], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))

        f1 += f2
        scope = SortedSet([Variable(0, 2, 'D'), Variable(2, 3, 'C')])
        self.assertEqual(scope, f1.scope)
        t3 = np.array(

                [
                    [11, 23, 35],
                    [12, 24, 36]
                ]
        )
        nptest.assert_almost_equal(t3, f1.table)
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))

    def test_binary_magic_sub(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C'), Variable(1, 2, 'C')], [-1.0, -2.0, -3.0, 4.0, 5.0, 6.0], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        f3 = f1 - f2
        print("\tstr={}, repr={}".format(str(f3), repr(f3)))

        scope3 = SortedSet( [Variable(0,2, 'D'), Variable(1,2, 'C'), Variable(2,3, 'C')] )
        self.assertEqual( scope3, f3.scope)

        t1 = np.array( [ [0.1, 0.3, -0.5], [0.2, -0.4, -0.6]] )
        t2 = np.array( [[-1, -3, 5], [-2, 4, 6]] )
        nptest.assert_almost_equal(t1, f1.table)
        nptest.assert_almost_equal(t2, f2.table)

        t3 = np.array(
            [
                [
                    [1.1, 3.3, -5.5],
                    [2.1, -3.7, -6.5]
                ],

                [
                    [1.2, 2.6, -5.6],
                    [2.2, -4.4, -6.6]
                ]
            ]
        )
        nptest.assert_almost_equal(t3, f3.table)

    def test_binary_magic_rsub(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [1, 2, 3, 4, 5, 6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = 1 - f1
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        self.assertEqual(f1.scope, f2.scope)
        t3 = np.array(
            [
                [1-1, 1-3, 1-5],
                [1-2, 1-4, 1-6]
            ]
        )

        nptest.assert_almost_equal(f2.table, t3)
        self.assertNotEqual(f1.fid, f2.fid)

    def test_binary_magic_sub_const(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [1, 2, 3, 4, 5, 6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = f1 - 1
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        self.assertEqual(f1.scope, f2.scope)
        t3 = np.array(
            [
                [1-1, 3-1, 5-1],
                [2-1, 4-1, 6-1]
            ]
        )
        nptest.assert_almost_equal(f2.table, t3)
        self.assertNotEqual(f1.fid, f2.fid)

    def test_binary_magic_isub_const(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [1, 2, 3, 4, 5, 6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f1 -= 1
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        t3 = np.array(
            [
                [1-1, 3-1, 5-1],
                [2-1, 4-1, 6-1]
            ]
        )
        nptest.assert_almost_equal(f1.table, t3)

    def test_binary_magic_isub_expand(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C'), Variable(1, 2, 'C')], [-1.0, -2.0, -3.0, 4.0, 5.0, 6.0], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))

        f1 += f2
        scope = SortedSet([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(2, 3, 'C')])
        self.assertEqual(scope, f1.scope)
        t3 = np.array(
            [
                [
                    [-0.9, -2.7, 4.5],
                    [-1.9, 4.3, 5.5]
                ],

                [
                    [-0.8, -3.4, 4.4],
                    [-1.8, 3.6, 5.4]
                ]
            ]
        )
        nptest.assert_almost_equal(t3, f1.table)
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))

    def test_binary_magic_isub_sub(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [1, 2, 3, 4, 5, 6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C')], [10.0, 20.0, 30.0], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))

        f1 += f2
        scope = SortedSet([Variable(0, 2, 'D'), Variable(2, 3, 'C')])
        self.assertEqual(scope, f1.scope)
        t3 = np.array(

                [
                    [11, 23, 35],
                    [12, 24, 36]
                ]
        )
        nptest.assert_almost_equal(t3, f1.table)
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))


    def test_binary_magic_mul(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [1, 2, 3, 4, 5, 6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C'), Variable(1, 2, 'C')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        f3 = f1 * f2
        print("\tstr={}, repr={}".format(str(f3), repr(f3)))

        scope3 = SortedSet([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(2, 3, 'C')])
        self.assertEqual(scope3, f3.scope)

        t1 = np.array([[1, 3, 5], [2, 4, 6]])
        t2 = np.array([[10, 30, 50], [20, 40, 60]])
        nptest.assert_almost_equal(t1, f1.table)
        nptest.assert_almost_equal(t2, f2.table)

        t3 = np.array(
            [
                [
                    [1 * 10, 3 * 30 , 5 * 50],
                    [1 * 20, 3 * 40, 5 * 60]
                ],

                [
                    [2 * 10, 4 * 30, 6 * 50],
                    [2 * 20, 4 * 40, 6 * 60]
                ]
            ]
        )
        nptest.assert_almost_equal(t3, f3.table)

    def test_binary_magic_rmul(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = 2 * f1
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        self.assertEqual(f1.scope, f2.scope)
        self.assertNotEqual(f1.fid, f2.fid)

    def test_binary_magic_mul_const(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = f1 * 5
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        self.assertEqual(f1.scope, f2.scope)
        self.assertNotEqual(f1.fid, f2.fid)
        t3 = np.array(
            [[0.1 * 5, 0.3 * 5, -0.5 * 5],
            [0.2 * 5, -0.4 * 5, -0.6 * 5]]
        )
        nptest.assert_almost_equal(f2.table, t3)

    def test_binary_magic_imul_const(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f1 *= 10
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        t2 = np.array(
            [
                [0.1*10, 0.3*10, -0.5*10],
                [0.2*10, -0.4*10, -0.6*10]
            ]
        )
        nptest.assert_almost_equal(t2, f1.table)

    def test_binary_magic_imul_expand(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [0.1, 0.2, 0.3, -0.4, -0.5, -0.6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C'), Variable(1, 2, 'C')], [2, 4, 6, 8, 10, 12], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))

        f1 *= f2
        scope = SortedSet([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(2, 3, 'C')])
        self.assertEqual(scope, f1.scope)
        t3 = np.array(
            [
                [
                    [0.1 * 2, 0.3 * 6, -0.5 * 10],
                    [0.1 * 4, 0.3 * 8, -0.5 * 12],
                ],

                [
                    [0.2 * 2, -0.4 * 6, -0.6 * 10],
                    [0.2 * 4, -0.4 * 8, -0.6 * 12]
                ]
            ]
        )
        nptest.assert_almost_equal(t3, f1.table)
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))

    def test_binary_magic_imul_sub(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [1, 2, 3, 4, 5, 6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C')], [10.0, 20.0, 30.0], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))

        f1 *= f2
        scope = SortedSet([Variable(0, 2, 'D'), Variable(2, 3, 'C')])
        self.assertEqual(scope, f1.scope)
        t3 = np.array(

            [
                [1*10, 3 *20, 5*30],
                [2*10, 4 * 20, 6*30]
            ]
        )
        nptest.assert_almost_equal(t3, f1.table)
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))

    def test_binary_magic_div(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C'), Variable(1, 2, 'C')], [1, 2, 3, 4, 5, 6], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        f3 = f1 / f2
        print("\tstr={}, repr={}".format(str(f3), repr(f3)))

        scope3 = SortedSet([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(2, 3, 'C')])
        self.assertEqual(scope3, f3.scope)

        t1 = np.array([[10, 30, 50], [20, 40, 60]])
        t2 = np.array([[1, 3, 5], [2, 4, 6]])
        nptest.assert_almost_equal(t1, f1.table)
        nptest.assert_almost_equal(t2, f2.table)

        t3 = np.array(
            [
                [
                    [10/1, 30/3, 50/5],
                    [10/2, 30/4, 50/6]
                ],

                [
                    [20/1, 40/3, 60/5],
                    [20/2, 40/4, 60/6]
                ]
            ]
        )
        nptest.assert_almost_equal(t3, f3.table)

    def test_binary_magic_rdiv(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = 120 /  f1
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        self.assertEqual(f1.scope, f2.scope)
        self.assertNotEqual(f1.fid, f2.fid)
        t3 = np.array(
            [
                [120/10, 120/30, 120/50],
                [120/20, 120/40, 120/60]
            ]
        )
        nptest.assert_almost_equal(t3, f2.table)

    def test_binary_magic_div_const(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = f1 / 10
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))
        self.assertEqual(f1.scope, f2.scope)
        self.assertNotEqual(f1.fid, f2.fid)
        t3 = np.array(
            [
                [10/10, 30/10, 50/10],
                [20/10, 40/10, 60/10]
            ]
        )
        nptest.assert_almost_equal(t3, f2.table)

    def test_binary_magic_idiv_const(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f1 /= 10
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        t2 = np.array(
            [
                [10/10, 30/10, 50/10],
                [20/10, 40/10, 60/10]
            ]
        )
        nptest.assert_almost_equal(t2, f1.table)

    def test_binary_magic_idiv_expand(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C'), Variable(1, 2, 'C')], [1, 2, 3, 4, 5, 6], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))

        f1 /= f2
        scope = SortedSet([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(2, 3, 'C')])
        self.assertEqual(scope, f1.scope)
        t3 = np.array(
            [
                [
                    [10/1, 30/3, 50/5],
                    [10/2, 30/4, 50/6]
                ],

                [
                    [20/1, 40/3, 60/5],
                    [20/2, 40/4, 60/6]
                ]
            ]
        )
        nptest.assert_almost_equal(t3, f1.table)
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))

    def test_binary_magic_idiv_sub(self):
        print(self.id())
        f1 = Factor([Variable(2, 3, 'C'), Variable(0, 2, 'D')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = Factor([Variable(2, 3, 'C')], [1, 2, 3], 'U')
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))

        f1 /= f2
        scope = SortedSet([Variable(0, 2, 'D'), Variable(2, 3, 'C')])
        self.assertEqual(scope, f1.scope)
        t3 = np.array(

            [
                [10/1, 30/2, 50/3],
                [20/1, 40/2, 60/3]
            ]
        )
        nptest.assert_almost_equal(t3, f1.table)
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))

    def test_sum_marginal_all_ip(self):
        print(self.id())
        f1 = Factor([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(2, 2, 'C')], [1, 2, 3, 4, 5, 6, 7, 8], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        eliminator = SortedSet([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(2, 2, 'C')])
        f1 = f1.sum_marginal(eliminator)        # in-place still catch returned reference (must when it returns float)
        nptest.assert_almost_equal(f1, 36)      # in-place means no building Factor; np.array can be created internally
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))

    def test_sum_marginal_all_noip(self):
        print(self.id())
        f1 = Factor([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(2, 2, 'C')], [1, 2, 3, 4, 5, 6, 7, 8], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        eliminator = SortedSet([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(2, 2, 'C')])
        f2 = f1.sum_marginal(eliminator, inplace=False)
        nptest.assert_almost_equal(f2, 36)
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        print("\tstr={}, repr={}".format(str(f2), repr(f2)))

    def test_sum_marginal_none(self):
        print(self.id())
        f1 = Factor([Variable(0, 2, 'D'), Variable(1, 2, 'C'), Variable(2, 2, 'C')], [1, 2, 3, 4, 5, 6, 7, 8], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        eliminator = SortedSet([Variable(3, 2, 'C'), Variable(4, 2, 'C'), Variable(5, 2, 'D')])
        f1.sum_marginal(eliminator)
        t = np.array(
            [
                [
                    [1, 2],
                    [3, 4]
                ],

                [
                    [5, 6],
                    [7, 8]
                ]
            ]
        )
        nptest.assert_almost_equal(f1.table, t)
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))

    def test_sum_marginal_ip1(self):
        print(self.id())
        f1 = Factor([Variable(0, 2, 'D'), Variable(1, 3, 'C')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f1.sum_marginal(SortedSet([Variable(0, 2, 'D') ]), inplace=True)
        t2 = np.array(
            [10+40, 20+50, 30+60]
        )
        nptest.assert_almost_equal(f1.table, t2)

    def test_sum_marginal_ip2(self):
        f1 = Factor([Variable(0, 2, 'D'), Variable(1, 3, 'C')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f1.sum_marginal(SortedSet([Variable(1, 3, 'C') ]), inplace=True)
        t2 = np.array(
            [60, 150]
        )
        nptest.assert_almost_equal(f1.table, t2)

    def test_sum_marginal_noip1(self):
        print(self.id())
        f1 = Factor([Variable(0, 2, 'D'), Variable(1, 3, 'C')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = f1.sum_marginal(SortedSet([Variable(0, 2, 'D') ]), inplace=False)
        t2 = np.array(
            [10+40, 20+50, 30+60]
        )
        nptest.assert_almost_equal(f2.table, t2)

    def test_sum_marginal_noip2(self):
        f1 = Factor([Variable(0, 2, 'D'), Variable(1, 3, 'C')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f2 = f1.sum_marginal(SortedSet([Variable(1, 3, 'C') ]), inplace=False)
        t2 = np.array(
            [60, 150]
        )
        nptest.assert_almost_equal(t2, f2.table)

    def test_max_marginal(self):
        print(self.id())
        f1 = Factor([Variable(0, 2, 'D'), Variable(1, 3, 'C')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f1.max_marginal(SortedSet([Variable(0, 2, 'D')]), inplace=True)
        t2 = np.array(
            [40, 50, 60]
        )
        nptest.assert_almost_equal(f1.table, t2)

    def test_min_marginal(self):
        print(self.id())
        f1 = Factor([Variable(0, 2, 'D'), Variable(1, 3, 'C')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f1.min_marginal(SortedSet([Variable(0, 2, 'D') ]), inplace=True)
        t2 = np.array(
            [10, 20, 30]
        )
        nptest.assert_almost_equal(f1.table, t2)

    def test_lse_marginal(self):
        print(self.id())
        f1 = Factor([Variable(0, 2, 'D'), Variable(1, 3, 'C')], [10, 20, 30, 40, 50, 60], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))
        f1.lse_marginal(SortedSet([Variable(0, 2, 'D')]), inplace=True)
        t2 = np.array(
            [np.log(np.exp(10)+np.exp(40)), np.log(np.exp(20)+np.exp(50)), np.log(np.exp(30)+np.exp(60))]
        )
        nptest.assert_almost_equal(f1.table, t2)
    def test_lse_pnorm_marginal(self):
        print(self.id())
        f1 = Factor([Variable(0, 2, 'D'), Variable(1, 3, 'C')], [1,2,3,4,5,6], 'U')
        print("\tstr={}, repr={}".format(str(f1), repr(f1)))

        f1.lse_pnorm_marginal(SortedSet([Variable(0, 2, 'D')]), p=3, inplace=True)
        t2 = np.array(
            [np.log(np.exp(1*3)+np.exp(4*3)), np.log(np.exp(2*3)+np.exp(5*3)), np.log(np.exp(3*3)+np.exp(6*3))]
        )
        t2 /= 3.0
        nptest.assert_almost_equal(f1.table, t2)


if __name__ == "__main__":
    unittest.main()
