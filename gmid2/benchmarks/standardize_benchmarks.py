import os

from gmid2.global_constants import *
from gmid2.basics.uai_files import standardize_util, rescale_util


def run(path_to_read, weight=1):
    for f in os.listdir(path_to_read):
        if f.endswith(".uai"):
            file_name = os.path.join(path_to_read, f).replace(".uai", "")
            rescale_util(file_name, weight)

if __name__ == "__main__":
    run(os.path.join(BENCHMARK_DIR, "sysadmin_pomdp"), 1)