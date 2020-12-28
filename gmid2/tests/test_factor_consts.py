import unittest
import os
from sortedcontainers import SortedSet, SortedDict, SortedList
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
import numpy.testing as nptest

from gmid2.global_constants import *
from gmid2.basics.factor import *
from gmid2.inference.pgm_bte import *

class FactorOpsTest(unittest.TestCase):
    # def _test_lse1(self):
    #     x = [Variable(0, 2, 'C'), Variable(1, 3, 'C')]
    #     table =  [np.log(1.0), np.log(3.0), np.log(5.0), np.log(2.0), np.log(4.0), np.log(6.0)]
    #     F0 = Factor(x, table)
    #     print(F0)
    #     F0.iexp()
    #     print(F0)
    #     F0.ilog()
    #     print(F0)
    #
    # def _test_lse2_elim_one_var_ip(self):
    #     x = [Variable(0, 2, 'C'), Variable(1, 3, 'C')]
    #     table =  [np.log(1.0), np.log(3.0), np.log(5.0), np.log(2.0), np.log(4.0), np.log(6.0)]
    #     F0 = Factor(x, table)
    #     print(F0)
    #     F0.lse_marginal(SortedSet([x[0]]), inplace=True)
    #     F1 = F0.exp()
    #     print(F1)
    #     nptest.assert_almost_equal(F1.table, np.array( [3,  7, 11 ]))
    #
    # def _test_lse2_elim_one_var(self):
    #     x = [Variable(0, 2, 'C'), Variable(1, 3, 'C')]
    #     table =  [np.log(1.0), np.log(3.0), np.log(5.0), np.log(2.0), np.log(4.0), np.log(6.0)]
    #     F0 = Factor(x, table)
    #     print(F0)
    #     F1 = F0.lse_marginal(SortedSet([x[0]]), inplace=False)
    #     F1.iexp()
    #     print(F1)
    #     nptest.assert_almost_equal(F1.table, np.array( [3,  7, 11 ]))
    #
    # def _test_lse3_elim_all_vars(self):
    #     x = [Variable(0, 2, 'C'), Variable(1, 3, 'C')]
    #     table =  [np.log(1.0), np.log(3.0), np.log(5.0), np.log(2.0), np.log(4.0), np.log(6.0)]
    #     F0 = Factor(x, table)
    #     print(F0)
    #     F0.lse_marginal(SortedSet([x[0], x[1]]), inplace=True)
    #     F1 = F0.exp()
    #     print(F1)
    #
    # def _test_lsepnorm_2_elim_one_var_ip(self):
    #     x = [Variable(0, 2, 'C'), Variable(1, 3, 'C')]
    #     table =  [np.log(1.0), np.log(3.0), np.log(5.0), np.log(2.0), np.log(4.0), np.log(6.0)]
    #     p = 1.0/0.5     # inverse of weight
    #     F0 = Factor(x, table)
    #     print(F0)
    #     F1 = F0.lse_pnorm_marginal(SortedSet([x[0]]), p, inplace=False)
    #     print(F1)
    #     # nptest.assert_almost_equal(F1.table, np.array( [3,  7, 11 ]))
    #
    #     # Factor0(0[2], 1[3]): = [0.         1.09861229 1.60943791 0.69314718 1.38629436 1.79175947]
    #     # Factor2(1[3]): = [0.80471896 1.60943791 2.05543693]
    #
    # def test_add_factor_constant(self):
    #     x = [Variable(0, 2, 'C'), Variable(1, 3, 'C')]
    #     table =  [np.log(1.0), np.log(3.0), np.log(5.0), np.log(2.0), np.log(4.0), np.log(6.0)]
    #     F0 = Factor(x, table)
    #     F0.iexp()
    #     print("F0 + 1.0 = {}".format(str(F0 + 1.0)))
    #     print("1.0 + F0 = {}".format(str(1.0+F0)))
    #
    #     print("F0 * 2.0 = {}".format(str(F0 * 2.0)))
    #     print("2.0 * F0 = {}".format(str(2.0 * F0)))
    #
    #     print("F0 - 2.0 = {}".format(str(F0 - 2.0)))
    #     print("2.0 - F0 = {}".format(str(2.0 - F0)))        # this is done by broadcasting
    #
    #     F1 = F0 + 1
    #     F1.sum_marginal(F1.vars, inplace=True)
    #     print(F1)   # 27 = 21  + 6      1 is added to all elements

    # def test_entropy(self):
    def test_rmul(self):
        x = [Variable(0, 2, 'C'), Variable(1, 3, 'C')]
        table =  [np.log(1.0), np.log(3.0), np.log(5.0), np.log(2.0), np.log(4.0), np.log(6.0)]
        F0 = Factor(x, table)
        F1 = 2.0 * F0
        print(F1)


