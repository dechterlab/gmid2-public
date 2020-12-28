PRJ_PATH = "/home/junkyul/conda/gmid2"
import sys
sys.path.append(PRJ_PATH)

import os
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from gmid2.global_constants import *
from gmid2.basics.uai_files import read_limid


def read_instance_stat(path, f):
    print("\nSTART {}".format(f))
    file_name = os.path.join(path, f.replace(".uai", ""))
    file_info = read_limid(file_name, skip_table=False)
    nvar = file_info.nvar
    nchance = file_info.nchance
    ndec = file_info.ndec

    nfunc = file_info.nfunc
    nprob = file_info.nprob
    nutil = file_info.nutil
    nblocks = file_info.nblock

    scopes = file_info.scopes
    max_scope = 0
    for sc in scopes:
        if len(sc) > max_scope:
            max_scope = len(sc)

    domains = file_info.domains
    max_domain = max(domains)

    print(f)
    print("n\t\t{}\nc\t\t{}\nd\t\t{}".format(nvar, nchance, ndec))
    print("f\t\t{}\np\t\t{}\nu\t\t{}".format(nfunc, nprob, nutil))
    print("k\t\t{}".format(max_domain))
    print("s\t\t{}".format(max_scope))
    print("END {}".format(f))

if __name__ == "__main__":
    path = os.path.join(BENCHMARK_DIR, "sysadmin_pomdp")
    for f in os.listdir(path):
        if f.endswith(".uai"):
            read_instance_stat(path, f.replace(".uai", ""))
