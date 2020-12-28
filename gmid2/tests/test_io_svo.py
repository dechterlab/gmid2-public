import unittest
import os
from gmid2.basics.uai_files import *

class TestSvo(unittest.TestCase):
    def test_write_read(self):
        svo = [
            [12, 8, 11, 17],
            [0, 9, 10, 7, 2, 13, 3, 16],
            [15],
            [0, 4, 2, 3, 14]
        ]
        widths = [4, 5, 1, 2]
        # if DEBUG__:       # jensen
        #     if sid == 0:
        #         bw_ordering, iw = [12, 8, 11, 17], 4
        #     if sid == 1:
        #         bw_ordering, iw = [0, 9, 10, 7, 2, 13, 3, 16], 5
        #     if sid == 2:
        #         bw_ordering, iw = [15], 1
        #     if sid == 3:
        #         bw_ordering, iw = [0, 4, 2, 3, 14], 2
        file_name = os.path.join(os.path.join(os.getcwd(), "test_data"), "jensen.svo")
        write_svo(file_name, svo, widths)


        file_info = FileInfo()
        file_info = read_svo(file_name, file_info)
        svo_read = file_info.svo
        widths_read = file_info.widths

        self.assertEqual(svo, svo_read)
        self.assertEqual(widths, widths_read)

        print(svo_read)
        print(widths_read)