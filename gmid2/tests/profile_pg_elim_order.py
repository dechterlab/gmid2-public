import os
from sortedcontainers import SortedSet, SortedDict, SortedList
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.basics.uai_files import read_limid, FileInfo, read_vo
from gmid2.basics.directed_network import DecisionNetwork
from gmid2.basics.undirected_network import PrimalGraph, greedy_variable_order
from gmid2.basics.undirected_network import get_induced_width_from_ordering, iterative_greedy_variable_order
from gmid2.global_constants import *

import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "ID_from_BN_0_w33d11")
file_info = read_limid(file_name, skip_table=True)
file_name = file_name + ".vo"
file_info = read_vo(file_name, file_info)

pg = PrimalGraph()
pg.build_from_scopes(file_info.scopes, file_info.var_types)
pvo = []
for each_block in file_info.blocks:  # blocks of constrained elim order
    pvo.append([pg.vid2nid[vid] for vid in each_block])

pr.enable()
ordering, iw = iterative_greedy_variable_order(primal_graph=pg, iter_limit=2000, pvo=pvo, pool_size=16)
print("found_iw:{}".format(iw))
pr.disable()
s= io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())



