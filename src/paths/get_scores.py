""" Scoring paths between entities """
# -*- coding: utf-8 -*-
import os
import json
from math import log
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from settings import FOLDER_PATH, NB_STATEMENT_DBPEDIA_GRAPHDB
from src.graphdb import ASK_TRIPLE_IN_KG, HEADERS_ASK, main_graphdb, \
    QUERY_PRED_COUNT, HEADERS_SELECT
from src.paths.get_primitives import main_primitives

def init_path_nb(paths: pd.DataFrame):
    """ Reset path numbers"""
    old_ind_to_new = {val: i for i, val in \
        enumerate(sorted(paths.path.unique()))}
    paths["new_path_index"] = paths.path.apply(lambda x: old_ind_to_new[x])
    return paths


def get_direction(row: pd.core.series.Series):
    """ Add if it's a reversed edge or not
    start = s, pred = p, end = e
    if not reversed -> (s, p, e) in KG
    if reversed -> (e, p, s) in KG """
    query = ASK_TRIPLE_IN_KG.replace("sub_uri", row.start) \
        .replace("pred_uri", row.pred) \
            .replace("obj_uri", row.end)
    res = main_graphdb(query=query, headers=HEADERS_ASK, type_o="json")
    row["order"] = "normal" if res["boolean"] else "reversed"
    return row


def load_cached_data(json_path):
    """ Retrieving cached data or returning empty dict """
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as openfile:
            return json.load(openfile)
    return {}


def get_weight(row, cached_data, metric):
    """ Retrieve weight of current sub path in the row
    Missing: cd - GP """

    if metric in ["in", "ou", "deg", "ns", "td", "so", "si", "sa"]:
        node = row.start if row.order == "normal" else row.end
        # node = row.start 
        return cached_data[node][metric]

    if metric == "informativeness":
        sub = row.start if row.order == "normal" else row.end
        obj = row.end if row.order == "normal" else row.start
        print(cached_data[sub])
        return (cached_data[sub]["pfo"][row.pred] + cached_data[obj]["pfi"][row.pred]) * cached_data[row.pred]["itf"] / 2

    raise ValueError(f"metric {metric} not implemented")


def extract_path_pattern(paths: pd.DataFrame):
    """ Counting patterns in graph """
    res_path = defaultdict(int)
    for path_nb in paths.path.unique():
        curr_df = paths[paths.path == path_nb]
        res_path["---".join(curr_df.pred.values)] += 1
    return res_path


def extract_diversity(paths: pd.DataFrame):
    labels = {index: set(paths[paths.new_path_index == index].pred.unique()) for index in paths.new_path_index.unique()}
    diversity = np.zeros((len(labels), len(labels)))

    nb_paths = np.max(paths.new_path_index.unique())
    for i in range(nb_paths):
        for j in range(i+1, nb_paths):
            diversity[i][j] = len(labels[i].intersection(labels[j])) / len(labels[i].union(labels[j]))
    return diversity


def get_scores(paths, metric):
    """ Main function """
    paths = init_path_nb(paths)

    print("Retrieving order of triples")
    tqdm.pandas()
    paths = paths.progress_apply(get_direction, axis=1)

    cached_data_path = os.path.join(
        FOLDER_PATH, "resources", "node_weights.json")
    cached_data = load_cached_data(json_path=cached_data_path)
    nodes = list(set(list(paths[paths.order == "normal"].start.unique()) + \
        list(paths[paths.order == "reversed"].end.unique())))
    # nodes = list(set(list(paths.start.unique()) + \
    #     list(paths.end.unique())))

    print("Retrieving info for nodes")
    new_nodes = [node for node in nodes if node not in cached_data]
    for i in tqdm(range(len(new_nodes))):
        cached_data[nodes[i]] = main_primitives(node=nodes[i])

    print("Retrieving info for predicates")
    preds = [x for x in list(paths.pred.unique()) if x not in cached_data]
    for i in tqdm(range(len(preds))):
        pred_count = main_graphdb(
                query=QUERY_PRED_COUNT.replace("pred_uri", preds[i]),
                headers=HEADERS_SELECT)["count"].values[0]
        cached_data[preds[i]] = {"itf": log(NB_STATEMENT_DBPEDIA_GRAPHDB/pred_count)}

    if metric == "pattern":
        patterns = extract_path_pattern(paths=paths)
        tot_pattern = sum(patterns.values())
        # print(paths.groupby('new_path_index')["pred"].transform(lambda x: log(tot_pattern / patterns["---".join(x)])))
        print(paths.groupby('new_path_index').agg({"pred": lambda x: log(tot_pattern / patterns["---".join(x)])}))
    else:
        paths["weight"] = paths.apply(lambda row: get_weight(row=row, cached_data=cached_data, metric=metric), axis=1)

        print(paths.groupby("new_path_index").agg({"weight": ["sum", "mean", "min", "max"]}))

    with open(cached_data_path, "w", encoding="utf-8") as openfile:
        json.dump(cached_data, openfile, indent=4)
    
    return paths


PATHS = pd.read_csv("./test_rank_paths/filtered_paths.csv")
PATHS = get_scores(PATHS, "pattern")
DIVERSITY = extract_diversity(paths=PATHS)
print(DIVERSITY)

# if not res_pred[pred]["itf"]:
            
