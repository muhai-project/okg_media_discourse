# -*- coding: utf-8 -*-
""" 
Input = tweets with their text
Output = features extracted
"""
import os
import argparse
import multiprocessing as mp
import psutil
import requests
import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
from transformers import pipeline
from src.helpers import read_csv, get_dask_df, check_args
from src.logger import Logger
from settings import SPARQL_ENDPOINT

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
PIPELINE_SENTIMENT = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL)

def run_ask_sparql_query(sparql_endpoint: str, roleset: str):
    """ ASK if roleset exists in dataset """
    query = """
    PREFIX pbdata: <https://w3id.org/framester/pb/pbdata/>

    ASK WHERE {{pbdata:<to-replace> ?p ?o} UNION {?s ?p pbdata:<to-replace>}}
    """.replace("<to-replace>", roleset)
    headers = {
        'Accept': 'application/sparql-results+json',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response=requests.get(sparql_endpoint, headers=headers, params={"query": query}, timeout=3600)

    try:
        return int(response.json()["boolean"])
    except Exception as _:
        return "error"


def add_frame_exist(pb_output, sparql_endpoint: str, cached: dict):
    """ Check if frames exist """
    if (not isinstance(pb_output, dict)) or \
        ("frameSet" not in pb_output):
        return [], cached
    res = []
    frames = [frame_info["frameName"] for frame_info in pb_output["frameSet"]]
    for frame in frames:
        if frame not in cached:
            cached[frame] = run_ask_sparql_query(sparql_endpoint, frame)
        res.append(cached[frame])
    return res, cached


def get_sentiment(text: str) -> dict:
    """ Roberta sentiment label with HF model """
    return PIPELINE_SENTIMENT(text)


def get_polarity_subjectivity(text: str) -> dict:
    """ w/ TextBlob """
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity}


def get_col_frame_exist(values: list[str], sparql_endpoint: str) -> list[int]:
    """ return list for frame_exist column """
    new_col, cached = [], {}
    for i in tqdm(range(len(values))):
        elt = values[i]
        res, cached = add_frame_exist(elt, sparql_endpoint, cached)
        new_col.append(res)
    return new_col


def get_col_post_level(values: list[str], func) -> list:
    """ return column for tweet level content """
    new_col = []
    content, score = None, None
    for i in tqdm(range(len(values))):
        curr_content = values[i]
        if curr_content != content:  # new tweet
            content = curr_content
            score = func(curr_content)
        new_col.append(score)
    return new_col


def extract_features(def_df_: dd.core.DataFrame, sparql_endpoint: str = SPARQL_ENDPOINT,
                     col_extract: str = "object") -> (pd.DataFrame, str):
    """ Adding features for building the KG """

    df_ = read_csv(def_df_[0].compute())

    print("Extracting sentiment")
    df_["sentiment"] = get_col_post_level(values=df_[col_extract].values, func=get_sentiment)

    print("Extracting polarity+subjectivity")
    df_["polarity_subjectivity"] = get_col_post_level(
        values=df_[col_extract].values, func=get_polarity_subjectivity)

    print("Extracting if frames exist")
    df_["frame_exist"] =  get_col_frame_exist(
        values=df_.propbank_output.values, sparql_endpoint=sparql_endpoint)

    return df_, def_df_[1]


def main(dfs: list[dd.core.DataFrame]) -> list:
    """ Main mp function """
    with mp.Pool(processes=psutil.cpu_count()) as pool:
        results = []
        for result in tqdm(pool.map(extract_features, dfs),
                           total=len(dfs)):
            results.append(result)

        pool.close()
        pool.join()
    return results

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', "--path", required=False,
                    help="path to .csv with data: (subject, 'description', content) per row")
    ap.add_argument('-f', "--folder", required=False,
                    help="folder with .csv files with preprocessed+pb output")
    # ap.add_argument('-c', "--column", required=True,
    #                 help="column to extract features from")
    # ap.add_argument('-s', "--sparql", required=True,
    #                 help="sparql endpoint")
    ap.add_argument('-o', "--output", required=False,
                    help="output folder")
    args_main = vars(ap.parse_args())

    check_args(args=args_main)
    DFS = get_dask_df(args=args_main)

    LOGGER = Logger()
    LOGGER.log_start(name="Extracting other features from tweets")
    # DF_MAIN = read_csv(args_main["path"])
    # DF_MAIN = extract_features(DF_MAIN, SPARQL_ENDPOINT, "object")

    if args_main["output"]:
        DFS = [x for x in DFS if not os.path.exists(
            os.path.join(
                args_main["output"],
                f"feat_{x[1].replace('.csv', '').split('_')[1]}.csv"))]

    # Running by batches of 100
    nb_batch = len(DFS)//100 + 1
    for index in range(nb_batch):
        print(f"Running batch {index+1}/{nb_batch} ({round(100*(index+1)/nb_batch, 2)}%)".upper())
        if index == nb_batch:
            CURR_DFS = DFS[(index+1)*100:]
        else:
            CURR_DFS = DFS[index*100:(index+1)*100]

        RESULTS = main(dfs=CURR_DFS)

        if args_main["output"]:
            for df_o, index in RESULTS:
                index = index.replace(".csv", "").split("_")[1]
                df_o.to_csv(os.path.join(args_main["output"], f"feat_{index}.csv"), encoding="utf8")

    LOGGER.log_end()
