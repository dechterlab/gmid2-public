import unittest
import os
from gmid2.basics.uai_files import FileInfo, read_limid


class ReadTest(unittest.TestCase):
    def setUp(self):
        self.file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "car")
        self.file_info = FileInfo()
        self.file_info.net_type = "ID"
        self.file_info.nvar = 6
        self.file_info.nfunc = 7
        self.file_info.nchance = 4
        self.file_info.ndec = 2
        self.file_info.nprob = 4
        self.file_info.nutil = 3
        self.file_info.var_types = ['D', 'C', 'C', 'C', 'C', 'D']
        self.file_info.domains = [2, 2, 2, 2, 2, 2]
        self.file_info.chance_vars = [1, 2, 3, 4]
        self.file_info.decision_vars = [0, 5]
        self.file_info.prob_funcs = [0, 1, 2, 3]
        self.file_info.util_funcs = [4, 5, 6]
        self.file_info.tables = [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.6, 0.4],
            [0.7, 0.3, 0.4, 0.6, 0.5, 0.5, 0.9, 0.1],
            [10.0, 20.0],
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 10.0, 30.0, 20.0]
        ]
        self.file_info.scopes = [[1], [2], [1, 0, 3], [2, 0, 4], [0], [1, 5], [2, 5]]
        self.file_info.factor_types = ['P', 'P', 'P', 'P', 'U', 'U', 'U']
        self.file_info.blocks = [[1, 2], [5], [4, 3], [0]]
        self.file_info.block_types = ['C', 'D', 'C', 'D']
        self.file_info.nblock = 4

    def tearDown(self):
        pass

    def test_read(self):
        file_info = read_limid(self.file_name, skip_table=False)
        self.assertEqual("car.uai", file_info.uai_file)
        self.assertEqual(self.file_info.net_type, file_info.net_type)
        self.assertEqual(self.file_info.nvar, file_info.nvar)
        self.assertEqual(self.file_info.nfunc, file_info.nfunc)
        self.assertEqual(self.file_info.nchance, file_info.nchance)
        self.assertEqual(self.file_info.ndec, file_info.ndec)
        self.assertEqual(self.file_info.nprob, file_info.nprob)
        self.assertEqual(self.file_info.nutil, file_info.nutil)
        self.assertEqual(self.file_info.var_types, file_info.var_types)
        self.assertEqual(self.file_info.domains, file_info.domains)
        self.assertEqual(self.file_info.chance_vars, file_info.chance_vars)
        self.assertEqual(self.file_info.decision_vars, file_info.decision_vars)
        self.assertEqual(self.file_info.prob_funcs, file_info.prob_funcs)
        self.assertEqual(self.file_info.util_funcs, file_info.util_funcs)
        self.assertEqual(self.file_info.scopes, file_info.scopes)
        self.assertEqual(self.file_info.factor_types, file_info.factor_types)
        self.assertEqual(self.file_info.blocks, file_info.blocks)
        self.assertEqual(self.file_info.block_types, file_info.block_types)
        self.assertEqual(self.file_info.nblock, file_info.nblock)
        self.assertEqual(len(self.file_info.tables), len(file_info.tables))
        for a, b in zip(self.file_info.tables, file_info.tables):
            for v1, v2 in zip(a, b):
                self.assertAlmostEqual(v1, v2)

        file_info.show_members()




if __name__ == "__main__":
    unittest.main()