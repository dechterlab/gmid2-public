import unittest
import os
import numpy.testing as nptest
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

import numpy as np
from gmid2.basics.uai_files import read_mmap, read_vo, read_sum, write_vo, read_mixed
from gmid2.basics.undirected_network import PrimalGraph, iterative_greedy_variable_order
from gmid2.basics.graphical_model import GraphicalModel
from gmid2.inference.bucket import bucket_tree_decomposition, mini_bucket_tree_decomposition, join_graph_decomposition
from gmid2.inference.pgm_bte import PgmBTE
from gmid2.inference.pgm_wmbmm import PgmWMBMM

class PgmBteForMiniBucketCarMixedTest(unittest.TestCase):
    def setUp(self):
        self.ibound = 4
    def test_run_pgm_mbte_carmixed(self):
        print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mixed"), "car.mixed")
        file_info = read_mixed(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_to_log()

        read_vo(file_name + ".vo", file_info)
        vid_elim_order = file_info.vo
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=self.ibound)

        # Message Passing
        bte = PgmBTE(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule(mbt)
        bte.init_propagate()
        bte.propagate_iter(bw_iter=True)
        bound = bte.bounds()
        print("mbe(i={}) {} (opt 70)".format(self.ibound, np.exp(bound)))

    def test_mpe_wmbmm(self):
        # print(self.id())
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data/mixed"), "car.mixed")
        file_info = read_mixed(file_name, skip_table=False)
        gm = GraphicalModel()
        gm.build(file_info)
        gm.convert_to_log()

        read_vo(file_name + ".vo", file_info)
        vid_elim_order = file_info.vo
        mbt = mini_bucket_tree_decomposition(gm, vid_elim_order, ibound=self.ibound)
        print(mbt)

        bte = PgmWMBMM(gm, vid_elim_order)
        bte.build_message_graph(mbt)
        bte.schedule()
        bte.init_propagate()
        bte.propagate_iter()
        bound = bte.bounds()
        print("wmbmm(i={}) {} (opt 70)".format(self.ibound, np.exp(bound)))

# wmbmm(i=1) 91.25050538564247 (opt 70)
# mbe(i=1) 88.000000364 (opt 70)

# wmbmm(i=2) 91.25050538564247 (opt 70)
# mbe(i=2) 88.000000364 (opt 70)

# wmbmm(i=3) 75.06626477123658 (opt 70)
# mbe(i=3) 1330.0000129250002 (opt 70)

# wmbmm(i=4) 70.00000072299999 (opt 70)
# mbe(i=4) 70.00000072299999 (opt 70)

# gdd(i=1)  1581.5539367438712
# gdd(i=3)  71.97637468594843