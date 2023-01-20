# -*- coding: utf-8 -*-
"""
Extracting further info on paths
"""
from collections import defaultdict
import numpy as np

def extract_path_pattern(paths):
    """ From the instantiated paths, extract the patterns """
    res_path, res_pred = defaultdict(int), defaultdict(int)
    for path_nb in paths.path.unique():
        curr_df = paths[paths.path == path_nb]
        res_path["---".join(curr_df.pred.values)] += 1
        for val in curr_df.pred.values:
            res_pred[val] += 1
    return res_path, res_pred
