import os
import csv


def combine_logs(domain, csv_stat, summary_path, csv_prefix, csv_postfix):
    combined_dict = {}
    combined_header1 = []
    combined_header2 = []
    with open(csv_stat, 'r') as fp:
        csv_reader = csv.reader(fp)
        header = next(csv_reader)
        combined_header1 += ['', "stats"]
        combined_header1 += ['']* (len(header)-2)
        combined_header2 += header
        for line in csv_reader:
            filename = line[0]
            combined_dict[filename] = {}
            combined_dict[filename]['stats'] = {}
            for ind, h in enumerate(header[1:], 1):
                combined_dict[filename]['stats'][h] = line[ind]

    summary_files = [f for f in os.listdir(summary_path) if f.startswith(csv_prefix) and f.endswith(csv_postfix)]
    for f in sorted(summary_files):
        with open(os.path.join(summary_path, f), 'r') as fp:
            alg_ibound = f.replace(csv_prefix, "").replace(csv_postfix, "")
            csv_reader = csv.reader(fp)
            header = next(csv_reader)
            combined_header1 += [alg_ibound]
            combined_header1 += ['']* (len(header)-3)
            combined_header2 += header[2:]
            for line in csv_reader:
                filename = line[0]

                try:
                    combined_dict[filename][alg_ibound] = {}
                except:
                    print('err')

                for ind, h in enumerate(header[1:], 1):
                    combined_dict[filename][alg_ibound][h] = line[ind]

    output_file = ".".join(["combined", domain])
    with open(output_file + ".csv", "w") as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow(combined_header1)
        csv_writer.writerow(combined_header2)

        for filename in sorted(combined_dict):
            row = [filename]
            for k1 in combined_header1:
                if k1 == "":
                    continue
                elif k1 == "stats":
                    for k2 in combined_header2[1:9]:
                        try:
                            row.append( combined_dict[filename]['stats'][k2] )
                        except:
                            row.append("")
                else:
                    for k2 in ["w", "w_list", "st", "build", "prop", "ub"]:
                        try:
                            row.append(combined_dict[filename][k1][k2])
                        except:
                            row.append("")
            csv_writer.writerow(row)


if __name__ == "__main__":
    domain = "synthetic_limid"     # SET
    date = "0609"               # EXP
    stat_path = os.path.join(os.getcwd(), "stats")
    summary_path = os.path.join(os.getcwd(), "summary")


    prefix = "summary."
    postfix = ".".join([date, domain, "csv"])
    combine_logs(domain, os.path.join(stat_path, "stats." + domain + ".csv"), summary_path, prefix, postfix)

