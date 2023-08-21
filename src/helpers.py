# -*- coding: utf-8 -*-
"""
Helpers functions for all scripts
"""
import os
import ast
import pickle
import spacy
import pandas as pd
from spacy.tokens import DocBin
import dask.dataframe as dd

def ent_to_uri(ent: spacy.tokens.Span) -> str:
    """ From entity return DBpedia URI """
    return ent._.dbpedia_raw_result["@URI"]


def get_spacy_docs_from_bytes(pkl_file: str) -> list:
    """ Read pkl file and rebuild spacy format """
    nlp = spacy.blank("en")
    with open(pkl_file, "rb") as openfile:
        bytes_data = pickle.load(openfile)
    doc_bin = DocBin().from_bytes(bytes_data)
    return list(doc_bin.get_docs(nlp.vocab))


def format_string_col(content: str):
    """ if str -> convert to Python structure, like dict or list """
    try:
        return ast.literal_eval(content)
    except Exception as _:
        return content


def read_csv(df_) -> pd.DataFrame:
    """ Read csv and convert columns to right format """
    if isinstance(df_, str):
        df_ = pd.read_csv(df_)
    df_ = df_[[c for c in df_.columns if not c.startswith("Unnamed: ")]]
    for col in ["propbank_output", "sentiment", "polarity_subjectivity",
                "frame_exist", "sent_mapping"]:
        if col in df_.columns:
            df_[col] = df_[col].apply(format_string_col)
    return df_


def check_args(args: dict):
    """ Checking whether `path` or `folder` key in args """
    if not (args.get("path") or args.get("folder")):
        raise ValueError("Cannot process further, either `path` arg" + \
            " or `folder` arg must be non empty")


def get_dask_df(args: dict) -> list[dd.core.DataFrame]:
    """ Read dask dataframes from either folder or path """
    if args["folder"]:
        dfs = sorted(os.listdir(args["folder"]))
        dfs = [(dd.read_csv(os.path.join(args["folder"], x ), on_bad_lines='skip', encoding="utf8"),
                x) for x in dfs]
    else:  # args_main["path"]
        dfs = [(dd.read_csv(args["path"], on_bad_lines='skip'), args["path"])]
    return dfs


def save_csv(args: dict, df_list: list[pd.DataFrame], prefix: str):
    """ Saving csv in folder """
    for index, df_o in enumerate(df_list):
        df_o.to_csv(os.path.join(args["output"], f"{prefix}_{index}.csv"))
