import unittest
import numpy as np
import numpy.testing as nptest
import os
from gmid2.global_constants import *
from gmid2.basics.factor import *
from sortedcontainers import SortedSet


class FactorArgmaxTest(unittest.TestCase):
    def _test1(self):
        v1, v2, v3 = Variable(0, 2, TYPE_CHANCE_VAR), Variable(1, 3, TYPE_CHANCE_VAR), Variable(2, 2, TYPE_DECISION_VAR)
        f = Factor([v1, v2], [1, 2, 3, 4, 5, 6])
        # max part  0   0   1   0/1     0   1
        print(f)
        # max for v1
        print(f.table[ (slice(None), 0) ])
        m = f.table[ (slice(None), 0)].argmax()
        print(f.table[ (slice(None), 0)].max())
        coord = np.unravel_index(m, (2, 3))
        print(coord)
        print(f[coord])
        print(f.argmax())
        print(f.table.argmax(axis=0))

    def test2(self):
        import itertools
        b = np.arange(2*3*4).reshape(2,3,4)
        b = np.array([1,4,2,3,  11,12,10,13,   4,9,1,2,    2,1,4,8,    22,31,12,3,  4,3,1,2]).reshape(2,3,4)
        for x,y,z in itertools.product( range(2), range(3), range(4)):
            print((x,y,z), b[x,y,z])
        print(b)

        print('swap 1, -1')
        c = b.swapaxes(1, -1)
        for x,y,z in itertools.product( range(2), range(4), range(3)):
            print((x,y,z), c[x,y,z])
        print(c)
        print(list(itertools.product( *[range(i) for i in [2,3,4]] )))

        d = c.reshape(-1)
        print(d)
        res = np.zeros(24)
        dec_dim = 3
        dec_max = -np.inf
        max_ind = -1
        for ind in range(24):
            if ind % dec_dim == 0:      # cycled one period so reset
                dec_max = -np.inf
                max_ind = -1
            if d[ind] > dec_max:
                res[ind] = 1
                dec_max = d[ind]
                res[max_ind] = 0
                max_ind = ind
        print(res)

# [0. 1. 0.
# 0. 1. 0.
# 0. 1. 0.
# 0. 1. 0.
# 0. 1. 0.
# 0. 1. 0.
# 0. 1. 0.
# 1. 0. 0.]

# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]
#
#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]]

# 9: 0, 2, 1        v1 0    v2 2    v3 1


# [[[ 0  4  8]
#   [ 1  5  9]
#   [ 2  6 10]
#   [ 3  7 11]]
#
# [[12 16 20]
#  [13 17 21]
#  [14 18 22]
#  [15 19 23]]]


#9: 0, 1, 2        v1 0     v3 1     v2 2       swapped