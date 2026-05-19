import os
import csv
import json
import yaml
import numpy as np
import scipy.stats
from tabulate import tabulate
from collections import OrderedDict

from sven.constant import VAL_SCENARIOS
from sven.constant import NOT_TRAINED, CWES_TRAINED

class SecEvalRun:
    TOP_K = [1, 5, 10]
    def __init__(self, eval_dir, eval_type, vul_types, split):
        self.results = OrderedDict()
        for cwe in vul_types:
            with open(os.path.join(eval_dir, cwe, 'result.jsonl')) as f:
                lines = f.readlines()
            for line in lines:
                j = json.loads(line)
                scenario = (cwe, j['scenario'])
                if eval_type in ('trained', 'trained_subset'):
                    if split == 'val' and scenario not in VAL_SCENARIOS:
                        continue
                    elif split == 'test' and scenario in VAL_SCENARIOS:
                        continue
                if scenario not in self.results:
                    self.results[scenario] = OrderedDict()
                self.results[scenario][j['control']] = j

                scores_path = os.path.join(eval_dir, cwe, j['scenario'], j['control']+'_scores.json')
                if os.path.exists(scores_path):
                    with open(scores_path) as f:
                        scores_j = json.load(f)
                    sorted_scores_j = list(sorted(scores_j.items(), reverse=True, key=lambda i:i[1]))
                    sorted_progs = list([i[0] for i in sorted_scores_j])
                    codeql_path = os.path.join(eval_dir, cwe, j['scenario'], j['control']+'_codeql.csv')
                    with open(codeql_path) as f:
                        reader = csv.reader(f)
                        vuls = set()
                        for row in reader:
                            vuls.add(row[4].replace('/', ''))
                    gens = set(scores_j.keys())
                    secs = gens - vuls
                    for k in self.TOP_K:
                        num_sec = len(secs & set(sorted_progs[:k]))
                        num_gen = min(k, len(gens))
                        j[f'sec_rate_{k}'] = num_sec / num_gen * 100

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h


class SecEval:
    KEYS = ['sec_rate', 'sec', 'total', 'non_parsed']
    available_eval_types = ['base','untrain']

    def __init__(self, eval_dir, split, eval_type):
        self.detail_results = OrderedDict()
        self.overall_results = OrderedDict()

        for et in self.available_eval_types:
            if et != eval_type and et == 'untrain':
                continue

            if et == 'base':
                evaled_scens = CWES_TRAINED
            elif et == 'untrain':
                evaled_scens = NOT_TRAINED
            else:
                assert False
            val_scens = VAL_SCENARIOS if et == 'base' else {}

            for cwe in evaled_scens:
                with open(os.path.join(eval_dir, et, cwe, 'result.jsonl')) as f:
                    lines = f.readlines()
                for line in lines:
                    j = json.loads(line)
                    scenario = (cwe, j['scenario'])
                    if split == 'val' and scenario not in val_scens:
                        continue
                    elif split == 'test' and scenario in val_scens:
                        continue
                    elif split == 'intersec' and cwe not in ['cwe-022', 'cwe-078', 'cwe-079', 'cwe-089']:
                        continue
                    elif split == 'diff' and cwe in ['cwe-022', 'cwe-078', 'cwe-079', 'cwe-089']:
                        continue
                    self.detail_results[scenario] = OrderedDict()
                    for key in self.KEYS:
                        if key == 'sec_rate':
                            self.overall_results['sec_rate'] = 0.0
                            if j['total'] != 0:
                                self.detail_results[scenario][key] = j['sec'] / j['total'] * 100
                            else:
                                self.detail_results[scenario][key] = 0.0
                        else:
                            if key not in self.overall_results:
                                self.overall_results[key] = 0
                            self.detail_results[scenario][key] = j[key]
                            self.overall_results[key] += j[key]
            self.overall_results['sec_rate'] = self.overall_results['sec'] / self.overall_results['total'] * 100

    def pretty_print(self, detail):
        table = []

        if detail:
            for scenario in self.detail_results:
                row = [scenario[0], scenario[1]]
                for key, value in self.detail_results[scenario].items():
                    row.append('{:.1f}'.format(value))
                table.append(row)

        row = ['overall', '']
        for key, value in self.overall_results.items():
            row.append('{:.1f}'.format(value))
        table.append(row)

        headers = ['cwe', 'scenario'] + list(self.overall_results.keys())
        print(tabulate(table, headers=headers, stralign='right', tablefmt='orgtbl'))

def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

class FuncEval:
    K = [1, 5, 10, 25, 50, 100]

    def __init__(self, eval_dir):
        self.pass_k = [[] for _ in range(len(self.K))]
        for fname in os.listdir(eval_dir):
            if not fname.endswith('.results.yaml'): continue
            with open(os.path.join(eval_dir, fname)) as f:
                res_data = yaml.load(f, Loader=yaml.CLoader)
            n, c = 0, 0
            for r in res_data['results']:
                n += 1
                if r['status'] == 'OK':
                    c += 1
            for i, k in enumerate(self.K):
                self.pass_k[i].append(pass_at_k(n, c, k))
        for i, k in enumerate(self.K):
            self.pass_k[i] = np.mean(self.pass_k[i])*100

    def pretty_print(self, detail):
        header, row = [], []
        for i, k in enumerate(self.K):
            header.append(f'pass@{k}')
            row.append('{:.1f}'.format(self.pass_k[i]))
        print(tabulate([row], headers=header, stralign='right', tablefmt='orgtbl'))

    def get_pass_k(self):
        res = OrderedDict()
        for i, k in enumerate(self.K):
            res[f'pass@{k}'] = self.pass_k[i]
        return res
