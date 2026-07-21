import argparse
import json

from pathlib import Path
from evaluator import create_results_table
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expe_folder', type=str, help="Folder containing the K(=5) experiments")
    _args = parser.parse_args()
    return _args


def update_summary(summ: defaultdict, res: dict, it: int):
    for keys, vals in res.items():
        for key, val in vals.items():
            for k, v in val.items():
                if isinstance(v, dict):
                    if it == 0:
                        summ[keys][key][k] = defaultdict(list)
                    for k_value, r_score in v.items():
                        summ[keys][key][k][k_value].append(r_score)
                else:
                    summ[keys][key][k].append(v)
    return summ


if __name__ == '__main__':
    args = parse_args()

    results_dir = Path(args.expe_folder)
    k_splits = [p for p in results_dir.iterdir() if p.is_dir()]
    k = int(results_dir.name.split("_")[0])
    assert len(k_splits) == k, f"Only {len(k_splits)}/{k} runs have been done"

    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, split in enumerate(k_splits):
        with open(f"{split}/results") as f:
            results = json.load(f)

        summary = update_summary(summary, results, i)

    print(create_results_table(summary, add_margin=True))
