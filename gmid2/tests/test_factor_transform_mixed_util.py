import unittest
import os
from sortedcontainers import SortedSet, SortedDict, SortedList
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
import numpy.testing as nptest

from gmid2.global_constants import *
from gmid2.basics.factor import *
from gmid2.inference.st_bte import transform_util_f, inv_transform_util_f

class FactorTransformMixed(unittest.TestCase):
    def test1(self):
        v1, v2 = Variable(0, 2, TYPE_CHANCE_VAR), Variable(1, 3, TYPE_CHANCE_VAR)
        f = Factor([v1, v2], [1, 2, 3, 4, 5, 6])
        fid = f.fid
        i1 = id(f)
        latent_var = Variable(3, 3, TYPE_CHANCE_VAR)
        transform_util_f(latent_var, f, 1)

        self.assertEqual(list(f.scope_vids), [0,1,3])
        self.assertEqual(f[0, 0, 0], np.log(latent_var.dim))
        self.assertEqual(f[0, 1, 0], np.log(latent_var.dim))
        self.assertEqual(f[0, 2, 0], np.log(latent_var.dim))
        self.assertEqual(f[1, 0, 0], np.log(latent_var.dim))
        self.assertEqual(f[1, 1, 0], np.log(latent_var.dim))
        self.assertEqual(f[1, 2, 0], np.log(latent_var.dim))
        self.assertEqual(f[0, 0, 1], 1)
        self.assertEqual(f[0, 1, 1], 2)
        self.assertEqual(f[0, 2, 1], 3)
        self.assertEqual(f[1, 0, 1], 4)
        self.assertEqual(f[1, 1, 1], 5)
        self.assertEqual(f[1, 2, 1], 6)
        self.assertEqual(fid, f.fid)
        self.assertEqual(i1, id(f))
        print(id(f))

        inv_transform_util_f(f, 1)
        self.assertEqual(np.ravel(f.table).tolist(), [1,2,3,4,5,6])
        print(f)