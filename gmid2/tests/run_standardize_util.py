import os
from gmid2.basics.uai_files import standardize_util, rescale_util, rescale_center_util, rescale_util_round

TEST_PATH = os.path.join(os.getcwd(), "test_data")
for f in os.listdir(TEST_PATH):
    if f.endswith(".uai") and "norm" not in f and "std" not in f and "mixed" not in f and "mmap" not in f and "hailfinder" not in f:
        file_name = os.path.join( TEST_PATH, f.replace(".uai", ""))
        rescale_util_round(file_name, s=10)
