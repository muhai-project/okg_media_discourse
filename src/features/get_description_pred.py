""" Getting distinct preds to convert csv triples to kg triples """
# -*- coding: utf-8 -*-
import os
import argparse
import multiprocessing as mp
import psutil
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm
from src.logger import Logger
from src.helpers import check_args, get_dask_df


def concat_des(df_list: list[pd.DataFrame]) -> pd.DataFrame:
    """ Concat text content of tweets """
    df_res = df_list[0]
    for curr_df in df_list[1:]:
        df_res = pd.concat([df_res, curr_df])
    return df_res


def concat_results(results: list[dict]) -> dict:
    """ Concat results from each pool """
    return {
        "preds": set(x for res in results for x in res["preds"]),
        "types": set(x for res in results for x in res["types"]),
        "shapes": [x for res in results for x in res["shapes"]],
        "content": [res["content"] for res in results],
    }


def print_info(data: dict):
    """ Print various info on the input triples """
    # SHAPES
    print(data["shapes"])

    # UNIQUE PREDS
    preds = data["preds"]
    print(f"{len(preds)} unique preds")
    dep_preds = [x for x in preds if x.startswith("dep_")]
    other_preds = list(set(preds).difference(set(dep_preds)))
    print(f"{len(dep_preds)} dependency predicates, {len(other_preds)} other predicates")
    print(f"{sorted(other_preds)}\n===============")

    # UNIQUE TYPES
    types = data["types"]
    print(f"{len(types)} unique object types")
    print(f"{sorted(list(types))}\n===============")

    # CONTENT
    print(data["content"][0].head(10))
    print([df.shape[0] for df in data["content"]])


def mp_func(input_df: dd.core.DataFrame) -> dict:
    """ Function for one df: retrieving descriptions"""
    input_df = input_df.compute()
    return {
        "preds": set(input_df["Predicate"].unique()),
        "types": set(input_df[input_df["Predicate"] == "rdf_type"]["Object"].unique()),
        "shapes": [input_df.shape[0]],
        "content": input_df[input_df["Predicate"] == "description"]
    }


def main(dfs: list[dd.core.DataFrame]) -> list:
    """ Returning main data """
    with mp.Pool(processes=psutil.cpu_count()) as pool:
        results = []
        for result in tqdm(pool.map(mp_func, dfs), total=len(dfs)):
            results.append(result)

        pool.close()
        pool.join()
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', "--folder", required=False,
                    help="folder of .csv files for data")
    ap.add_argument('-p', "--path", required=False,
                    help=".csv file with triple structure")
    # ap.add_argument('-cp', "--col_pred", required=True,
    #                 help="column corresponding to predicate")
    # ap.add_argument('-co', "--col_obj", required=True,
    #                 help="column corresponding to object")
    ap.add_argument('-o', "--output", required=False,
                    help="output folder for tweet description")
    ap.add_argument('-pr', "--print", required=False, default="0",
                    help="print info about triples")
    args_main = vars(ap.parse_args())

    check_args(args=args_main)

    DFS =  get_dask_df(args=args_main)

    LOGGER = Logger()
    LOGGER.log_start(name="Extracting descriptions of tweets")
    DATA = main(dfs=[x[0] for x in DFS])
    DATA = concat_results(results=DATA)

    LOGGER.log_end()

    if args_main["print"] == "1":
        print_info(data=DATA)

    if args_main["output"]:
        for i, df_o in enumerate(DATA["content"]):
            df_o.columns = ["subject", "predicate", "object"]
            df_o.to_csv(os.path.join(args_main["output"], f"des_{i}.csv"))
