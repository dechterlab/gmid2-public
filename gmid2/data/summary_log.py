import os
import sys
from gmid2.basics.uai_files import get_token
import csv
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4)

def summary_log(CASE, ibound, DATE, DOMAIN, path):
    prefix = ".".join([CASE, ibound, DATE, DOMAIN, ""])
    postfix = ".out"
    summary_dict = dict()
    for f in os.listdir(path):
        if f.startswith(prefix) and f.endswith(postfix):
            count = f.split(".")[-2]
            with open(os.path.join(path, f), 'r') as fp:
                token = get_token(fp)
                try:
                    algorithm_name = next(token)
                    ibound = next(token)
                except:
                    print(f)
                    continue
                w = -1
                w_list = []
                for word in token:
                    if word == "START":
# <<<<<<< HEAD
                        get_word = next(token)
                        if get_word == "SUBMODEL":
                            skip_word = next(token)
                            while skip_word != "st":
                                skip_word = next(token)
                            summary_dict[filename]['st'] = float(next(token))
                        elif get_word not in summary_dict:
                            filename = get_word
# =======
#                         filename = next(token)
#                         if filename not in summary_dict:
# >>>>>>> openlab
                            summary_dict[filename] = dict()
                            summary_dict[filename]['count'] = count
                            summary_dict[filename]['algorithm'] = algorithm_name
                            summary_dict[filename]['ibound'] = ibound
                    if word == "w":
                        temp = int(next(token))
                        w_list.append(temp)
                        if temp > w:
                            w = temp
                    if word == "build":
                        summary_dict[filename]["w"] = w
                        summary_dict[filename]["w_list"] = w_list
                        summary_dict[filename]["build"] = float(next(token))
                    if word == "prop":
                        summary_dict[filename]["prop"] = float(next(token))
                    if word == "ub":
                        try:
                            summary_dict[filename]["ub"] = float(next(token))
                        except:
                            summary_dict[filename]["ub"] = float('inf')


    # pp.pprint(summary_dict)
    if len(summary_dict) == 0:
        return

    output_file = ".".join(["summary", CASE, ibound, DATE, DOMAIN])
    with open(output_file + ".csv", "w") as fp:
        csv_writer = csv.writer(fp)
        header = ["file", "count", "w", "w_list", "st", "build", "prop", "ub" ]
        csv_writer.writerow(header)
        for filename in sorted(summary_dict):
            row = []
            for k in header:
                if k == "file":
                    row.append(filename)
                else:
                    try:
                        row.append(summary_dict[filename][k])
                    except:
                        row.append("")
            csv_writer.writerow(row)


if __name__ == "__main__":
    # SET = sys.argv[1]
    # CASE = sys.argv[2]
    # ibound = sys.argv[3]
    # EXP = sys.argv[4]
    # path = sys.argv[5]
    DATE = "0609"
    path = os.path.join( os.getcwd(), "logs_0609_synthetic")

    for CASE in ["STWMBMMBW"]:
        for DOMAIN in ["synthetic_limid"]:
            for ibound in ["10", "12", "14", "16", "18", "20"]:

                summary_log(CASE, ibound, DATE, DOMAIN, path)


