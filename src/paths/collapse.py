# -*- coding: utf-8 -*-
""" Collapsing paths between entities

Intuition: the (shortest) paths between two entities are often short (wrt # of nodes) but there are many of them. Some of these shortest paths between two same entities go through the same nodes and/or carry the same meaning 
--> trying to reduce the number of paths, and to find one representative for paths with the same meaning

Description of a .csv containing shortest path
- path: path number
- index_path: path triple number
- start, pred, end: subject, predicate, object of triple `index_path`
"""
import os
import json
import argparse
from tqdm import tqdm
import pandas as pd
from pyvis.network import Network

from src.graphdb import QUERY_PRED_COUNT, main_graphdb
from settings import FOLDER_PATH

def pre_process(node):
    """ URI > more human-readable """
    return node.split("/")[-1].replace('_', ' ')

def get_val_from_index(df_pd, col_number):
    """ Returning list of values from column number in index """
    return [x[col_number] for x in df_pd.index]

def build_summary_paths(grouped_path_seg, output_folder):
    """ Graph summary of shortest paths """
    nt_summary_graph = Network("850px", "850px", notebook=False,
                               directed=True, layout="hierarchical")
    curr_start, curr_pred, curr_end = "", "", ""
    grouped_path_seg = grouped_path_seg.sort_values(by=["index_path", "start", "end"])
    for index, row in grouped_path_seg.iterrows():
        node_label_1 = pre_process(row.start)
        node_label_2 = pre_process(row.end)
        if node_label_1 not in nt_summary_graph.nodes:
            nt_summary_graph.add_node(node_label_1, level=row.index_path, title=node_label_1)
        if node_label_2 not in nt_summary_graph.nodes:
            nt_summary_graph.add_node(node_label_2, level=row.index_path+1, title=node_label_2)

        if not (curr_start == row.start and curr_end == row.end) and index > 0:
            nt_summary_graph.add_edge(pre_process(curr_start), pre_process(curr_end),
                                      title=curr_pred, value=len(curr_pred.split("\n")))
            curr_pred = f"{row.pred}\n"
        else:
            curr_pred += f"{row.pred}\n"
        curr_start, curr_end = row.start, row.end
    nt_summary_graph.hrepulsion(node_distance=300)
    nt_summary_graph.show(os.path.join(output_folder, "summary_graph.html"))

def update_pred_count(pred_count_path, df_pd):
    """ Update cached values of predicate count in dataset
     """
    if os.path.exists(pred_count_path):
        with open(pred_count_path, encoding="utf-8") as openfile:
            pred_count = json.load(openfile)
    else:
        pred_count = {}

    to_update = [pred for pred in df_pd.pred.unique() if pred not in pred_count]
    print("Updating cached predicate count values")
    for i in tqdm(range(len(to_update))):
        curr_pred = to_update[i]
        df_out = main_graphdb(query=QUERY_PRED_COUNT.replace("pred_uri", curr_pred))
        pred_count[curr_pred] = str(df_out["count"].values[0])
    print("Finished updating cached predicate count values")

    df_pd["pred_count"] = df_pd["pred"].apply(lambda x: int(pred_count[x]))
    with open(pred_count_path, "w", encoding="utf-8") as openfile:
        json.dump(pred_count, openfile, indent=4)

    return df_pd

def select_representative_pred(preds):
    """ Given a list of predicates, choose one representative
    - Favouring dbo over others, like dbp (prefixes)
    - Checking frequency in DBpedia, taking the one with lower frequency 

    Element in `pred`: (pred_uri, # of times in dataset)"""
    dbo_paths, other_paths = [], []
    for (uri, count) in preds:
        if uri.startswith("http://dbpedia.org/ontology/"):
            dbo_paths.append((uri, count))
        else:
            other_paths.append((uri, count))

    return sorted(dbo_paths, key=lambda x: x[1], reverse=False) + \
        sorted(other_paths, key=lambda x: x[1], reverse=False)



def main(df_paths, output_folder):
    """ Grouping shortest paths """
    grouped = df_paths.groupby(['index_path', 'start', 'pred', 'end']).agg({'path': 'count'})
    grouped = grouped.sort_index(level=[0,1,3])
    grouped = grouped.reset_index(level=['index_path', 'start', 'pred', 'end'])
    grouped \
        .to_csv(os.path.join(output_folder, "grouped_path_segment.csv"))
    print(f"Grouped path segments saved to folder {output_folder}")
    build_summary_paths(grouped_path_seg=grouped, output_folder=output_folder)
    print(f"Paths summary visualisations saved to folder {output_folder}")
    grouped = update_pred_count(pred_count_path=os.path.join(FOLDER_PATH, "resources", "pred_count.json"),
                      df_pd=grouped)

    grouped_repr = df_paths.groupby(["start", "end"]).agg({'pred': 'nunique'})
    grouped_repr = grouped_repr.reset_index(level=["start", "end"])

    df_paths["pred_repr"] = ""
    for _, row in tqdm(grouped_repr.iterrows(), total=grouped_repr.shape[0]):

        representative = select_representative_pred(preds= \
            grouped[(grouped.start == row.start) & (grouped.end == row.end)][["pred", "pred_count"]].values)
        df_paths.loc[
            (df_paths["start"] == row.start) & (df_paths["end"] == row.end), "pred_repr"] = representative[0][0]

    df_paths = df_paths[[x for x in df_paths.columns if x != "Unnamed: 0"]]
    filtered_paths = df_paths[df_paths.pred == df_paths.pred_repr]
    
    len_path = df_paths.index_path.max() + 1
    grouped_paths = filtered_paths.groupby("path").agg({"index_path": "count"})
    selected_paths = grouped_paths[grouped_paths.index_path == len_path].index
    filtered_paths[filtered_paths.path.isin(selected_paths)].to_csv(os.path.join(output_folder, "filtered_paths.csv"))



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', "--paths", required=True,
                    help=".csv file with paths between two entities")
    ap.add_argument('-o', "--output_folder", required=True,
                    help="output folder to save html, csv etc")
    args_main = vars(ap.parse_args())

    DF_PATHS = pd.read_csv(args_main["paths"])
    main(df_paths=DF_PATHS, output_folder=args_main['output_folder'])
