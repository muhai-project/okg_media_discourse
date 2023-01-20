# -*- coding: utf-8 -*-
"""
Calculating metrics from GraphDB results
"""
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from settings import FOLDER_PATH

def extract_path_pattern(paths):
    """ From the instantiated paths, extract the patterns """
    res_path, res_pred = defaultdict(int), defaultdict(int)
    for path_nb in paths.path.unique():
        curr_df = paths[paths.path == path_nb]
        res_path["---".join(curr_df.pred.values)] += 1
        for val in curr_df.pred.values:
            res_pred[val] += 1
    return res_path, res_pred

def get_path_info(row, folder):
    """ Adding info on paths: # path pattern, # predicates involved """
    [start, end] = sorted([row.ent_1, row.ent_2])
    start = start.replace("http://dbpedia.org/resource/", "")
    end = end.replace("http://dbpedia.org/resource/", "")
    paths_csv = f"{FOLDER_PATH}/{folder}/{start}_{end}.csv"
    paths = pd.read_csv(paths_csv)

    res_path, res_pred = extract_path_pattern(paths)

    row["path_pattern"] = len(res_path)
    row["path_pred"] = len(res_pred)
    return row

def print_metrics(data, folder_paths):
    """ Getting mains metrics from data """

    label_to_metric = {
        "min": np.min, "mean": np.mean,
        "median": np.median, "max": np.max
    }
    print(f"# of pairs: {data.shape[0]}")
    spl = data[data.shortest_path_length.notnull()]

    tqdm.pandas()
    spl = spl.progress_apply(lambda row: get_path_info(row, folder=folder_paths), axis=1)

    print(f"# of pairs with path found: {spl.shape[0]}" + \
        f" ({round(100*spl.shape[0]/data.shape[0], 1)}%)")
    for (col, label) in [("shortest_path_length", "SPL"), ("nb_paths", "SPL paths"),
                         ("path_pattern", "Path Pattern"), ("path_pred", "Path pred")]:
        for metric in ["min", "mean", "median", "max"]:
            print(f"{metric.capitalize()} {label}: {label_to_metric[metric](spl[col].values)}")

if __name__ == '__main__':
    # python src/get_metrics graphdb_inequality.csv
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', "--file", required=True,
                    help=".csv file with info")
    ap.add_argument('-p', "--paths", required=True,
                    help="folder containing paths between entities")
    args_main = vars(ap.parse_args())

    DATA = pd.read_csv(args_main["file"])
    print_metrics(data=DATA, folder_paths=args_main["paths"])
