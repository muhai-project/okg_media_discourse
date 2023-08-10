# -*- coding: utf-8 -*-
""" 
Input = tweets with their text
Output = features extracted
"""
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
from transformers import pipeline
# from src.propbank_grammar.run_propbank_grammar import call_propbank_grammar
from src.helpers import read_csv
from src.logger import Logger

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
PIPELINE_SENTIMENT = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL)
HEADERS = {
    'Accept': 'application/sparql-results+json',
    'Content-Type': 'application/x-www-form-urlencoded'
}

def run_sparql_query(sparql_endpoint, roleset):
    """ ASK if roleset exists in dataset """
    query = """
    PREFIX pbdata: <https://w3id.org/framester/pb/pbdata/>

    ASK WHERE {{pbdata:<to-replace> ?p ?o} UNION {?s ?p pbdata:<to-replace>}}
    """.replace("<to-replace>", roleset)
    response=requests.get(sparql_endpoint, headers=HEADERS, params={"query": query}, timeout=3600)

    try:
        return int(response.json()["boolean"])
    except Exception as _:
        return "error"

def add_frame_exist(pb_output, sparql_endpoint, cached):
    """ Check if frames exist """
    if not isinstance(pb_output, dict):
        return [], cached
    res = []
    frames = [frame_info["frameName"] for frame_info in pb_output["frameSet"]]
    for frame in frames:
        if frame not in cached:
            cached[frame] = run_sparql_query(sparql_endpoint, frame)
        res.append(cached[frame])
    return res, cached


def get_sentiment(text):
    """ Roberta sentiment label with HF model """
    return PIPELINE_SENTIMENT(text)

def get_polarity_subjectivity(text):
    """ w/ TextBlob """
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity}

def get_col_frame_exist(values, sparql_endpoint):
    """ return list for frame_exist column """
    new_col, cached = [], {}
    for i in tqdm(range(len(values))):
        elt = values[i]
        res, cached = add_frame_exist(elt, sparql_endpoint, cached)
        new_col.append(res)
    return new_col

def get_col_post_level(values, func):
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

def extract_features(df_, sparql_endpoint, col_extract):
    """ Adding features for building the KG """

    # df_["sentiment"] = df_[col_extract].progress_apply(get_sentiment)
    # df_["polarity_subjectivity"] = df_[col_extract].progress_apply(get_polarity_subjectivity)

    print("Extracting sentiment")
    df_["sentiment"] = get_col_post_level(values=df_[col_extract].values, func=get_sentiment)

    print("Extracting polarity+subjectivity")
    df_["polarity_subjectivity"] = get_col_post_level(values=df_[col_extract].values, func=get_polarity_subjectivity)

    print("Extracting if frames exist")
    df_["frame_exist"] =  get_col_frame_exist(values=df_.propbank_output.values, sparql_endpoint=sparql_endpoint)

    return df_

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', "--path", required=True,
                    help="path to .csv with data: (subject, 'description', content) per row")
    ap.add_argument('-c', "--column", required=True,
                    help="column to extract features from")
    ap.add_argument('-s', "--sparql", required=True,
                    help="sparql endpoint")
    ap.add_argument('-o', "--output", required=False,
                    help="output csv")
    args_main = vars(ap.parse_args())

    LOGGER = Logger()

    LOGGER.log_start(name="Extracting other features from tweets")
    DF_MAIN = read_csv(args_main["path"])
    DF_MAIN = extract_features(DF_MAIN, args_main["sparql"],args_main["column"])

    print(DF_MAIN.head())
    if args_main["output"]:
        DF_MAIN.to_csv(args_main["output"])
    LOGGER.log_end()
