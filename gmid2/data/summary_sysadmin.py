import os
import re
import pprint
import csv

pp = pprint.PrettyPrinter(indent=4)


planner = 'snap'
pattern = re.compile(r'>>> SESSION END, total reward = (\S+)')
directory = os.path.join( os.getcwd(), planner)
# >>> SESSION END, total reward = 14582.75
# >>> SESSION END, total reward = 2545.0000000000005

results = {}
for f in sorted(os.listdir(directory)):
    domain, inst, time, _ = f.split('.')
    time = int(time.split('-')[-1])
    with open(os.path.join(directory, f), 'r') as fp:
        txt = fp.read()
        m = pattern.search(txt)
        if m != None:
            reward = float(m.group(1))
            if inst not in results:
                results[inst] = {}
            results[inst][time] = reward/40

pp.pprint(results)

with open(planner + ".csv", 'w') as fp:
    csv_writer = csv.writer(fp)
    csv_writer.writerow(['inst'] + [str(el) for el in range(3, 11)])
    for inst in sorted(results):
        row = [inst]
        for t in sorted(results[inst]):
            row.append(results[inst][t])
        csv_writer.writerow(row)
