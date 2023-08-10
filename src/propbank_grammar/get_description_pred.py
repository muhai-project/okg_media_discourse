""" Getting distinct preds to convert csv triples to kg triples """
# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
from src.logger import Logger

def concat_des(df_list):
    """ Concat text content of tweets """
    df_res = df_list[0]
    for curr_df in df_list[1:]:
        df_res = pd.concat([df_res, curr_df])
    return df_res

def main(df_list, col_pred, col_obj):
    """ Returning main data """
    return {
        "preds": set(x for df in df_list for x in df[col_pred].unique()),
        "types": set(x for df in df_list for x in df[df[col_pred] == "rdf_type"][col_obj].unique()),
        "shapes": [df.shape[0] for df in df_list],
        "content": concat_des(list(df[df[col_pred] == "description"] for df in df_list))
    }

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', "--folder", required=True,
                    help="folder of .csv files for data")
    ap.add_argument('-cp', "--col_pred", required=True,
                    help="column corresponding to predicate")
    ap.add_argument('-co', "--col_obj", required=True,
                    help="column corresponding to object")
    ap.add_argument('-o', "--output", required=False,
                    help="output .csv file for tweet description")
    args_main = vars(ap.parse_args())

    DF_LIST = os.listdir(args_main["folder"])
    DF_LIST = [pd.read_csv(os.path.join(args_main["folder"], x )) for x in DF_LIST]

    LOGGER = Logger()
    LOGGER.log_start(name="Extracting descriptions of tweets")
    DATA = main(df_list=DF_LIST, col_pred=args_main["col_pred"],
                col_obj=args_main["col_obj"])
    LOGGER.log_end()
    
    # SHAPES
    print(DATA["shapes"])

    # UNIQUE PREDS
    PREDS = DATA["preds"]
    print(f"{len(PREDS)} unique preds")
    DEP_PREDS = [x for x in PREDS if x.startswith("dep_")]
    OTHER_PREDS = list(set(PREDS).difference(set(DEP_PREDS)))
    print(f"{len(DEP_PREDS)} dependency predicates, {len(OTHER_PREDS)} other predicates")
    print(f"{sorted(OTHER_PREDS)}\n===============")

    # UNIQUE TYPES
    TYPES = DATA["types"]
    print(f"{len(TYPES)} unique object types")
    print(f"{sorted(list(TYPES))}\n===============")

    # CONTENT
    print(DATA["content"])
    if args_main["output"]:
        DATA["content"].columns = ["subject", "predicate", "object"]
        DATA["content"].to_csv(args_main["output"])
