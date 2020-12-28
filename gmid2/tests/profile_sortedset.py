from time import time
from sortedcontainers import SortedSet
import numpy as np

a = list( np.random.randint(0, 1000000, 10000) )

t0 = time()
for i in range(1000):
    A = SortedSet(a)
print("SortedSet:{}".format(time()-t0))

t0 = time()
for i in range(1000):
    B = set( sorted(a) )
print("sorted+set:{}".format(time()-t0))



