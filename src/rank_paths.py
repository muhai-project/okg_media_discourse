# -*- coding: utf-8 -*-
""" Ranking shortest paths given by GraphDB

Ideas
- Favour paths with entities that appear in the twitter graph
    * sum the number of times (same entity can be counted twice)
    * sum unique entities
- Using (graph or text) embedding similarity + threshold
- Extracted weighted graph with nodes = entities encountered during traversal, weights = number of predicates linking an entity to another in the graph (~path summary) """
import json
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

def update_count(former_val: int, type_count: str):
    """ Input
    - former_val: value to be updated
    - type_count: if 'ncount' increase by one, if 'unique' returns one """
    if type_count == "ncount":
        return former_val + 1
    if type_count == "unique":
        return 1
    return None

def get_matrix_entity_in_path(vocab: dict, paths: pd.DataFrame, type_count: str):
    """ Input
    - vocab: key = number string, value = corresponding (DBpedia) entity
    - paths: Output of query for shortest path, cf get_shortest_path.py, with following columns: `path`, `index_path`, `start`, `pred`, `end`
    Output
    - `type_count` in ['ncount', 'unique']. If ncount sum over path axis (same entity can be counted twice). If unique, one entity is only counted once, even if it appears several times in the paths
    - Sparse matrix M, # paths * size vocab. M[i,j] = number of times entity j appear in path j
    """
    possible_type_counts = ['ncount', 'unique']
    if type_count not in possible_type_counts:
        raise ValueError(f"`type_count` argument should be in {possible_type_counts}")

    entity_to_id = {val: key for key, val in vocab.items()}
    res = lil_matrix((np.max(paths.path.unique()) + 1, len(vocab) + 1), dtype=np.int8)
    sp_length = np.max(paths.index_path.unique())

    for _, row in tqdm(paths.iterrows(), total=paths.shape[0]):
        if row.start in entity_to_id:
            former_val = res[row.path, int(entity_to_id[row.start])]
            res[row.path, int(entity_to_id[row.start])] = \
                update_count(former_val, type_count)

        if row.index_path == sp_length:
            former_val = res[row.path, int(entity_to_id[row.end])]
            res[row.path, int(entity_to_id[row.end])] = \
                update_count(former_val, type_count)

    print(res.shape)
    print(res.sum(axis=1))
    print(res.sum(axis=1).shape)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', "--vocab", required=True,
                    help=".json file with vocab of entities")
    ap.add_argument('-p', "--paths", required=True,
                    help=".csv file with paths between two entities")
    ap.add_argument('-t', "--type_count", required=True,
                    help="type count for ranking")
    args_main = vars(ap.parse_args())

    with open(args_main["vocab"], "r", encoding="utf-8") as openfile:
        VOCAB = json.load(openfile)

    PATHS = pd.read_csv(args_main["paths"])
    get_matrix_entity_in_path(
        vocab=VOCAB, paths=PATHS, type_count=args_main["type_count"])
