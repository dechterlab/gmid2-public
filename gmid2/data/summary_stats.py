import os
from gmid2.basics.uai_files import get_token
import csv
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4)

def summary_stat(filepath, domain):
    stat_dict = {}
    with open(filepath, 'r') as fp:
        token = get_token(fp)
        for word in token:
            if word == "START":
                filename = next(token)
                stat_dict[filename] = {}
                next(token)     # consume one

                while True:
                    kk = next(token)
                    if kk in ["n", "c", "d", "f", "p", "u", "k", "s"]:
                        stat_dict[filename][kk] = next(token)
                    elif kk == "END":
                        break


    pp.pprint(stat_dict)

    output_file = ".".join(["stats", domain])
    with open(output_file + ".csv", "w") as fp:
        csv_writer = csv.writer(fp)
        header = ["file", "n", "c", "d", "f", "p", "u", "k", "s"]
        csv_writer.writerow(header)
        for filename in sorted(stat_dict):
            row = []
            for k in header:
                if k == "file":
                    row.append(filename)
                else:
                    try:
                        row.append(stat_dict[filename][k])
                    except:
                        row.append("-")
            csv_writer.writerow(row)


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "stats")
# <<<<<<< HEAD
    summary_stat(os.path.join(path, "sysadmin_pomdp.txt"), "sysadmin_pomdp")
# =======
#     summary_stat(os.path.join(path, "synthetic_limid.txt"), "synthetic_limid")
# >>>>>>> openlab


