import os
from gmid2.global_constants import BENCHMARK_DIR, TYPE_LIMID_NETWORK
from gmid2.basics.uai_files import change_file_type

def run(path_to_read):
    for f in os.listdir(path_to_read):
        if f.endswith(".uai"):
            file_name = os.path.join(path_to_read, f).replace(".uai", "")
            new_file_name = file_name + "_limid"
            change_file_type(file_name, new_file_name, TYPE_LIMID_NETWORK)


if __name__ == "__main__":
    run(os.path.join(BENCHMARK_DIR, "synthetic"))