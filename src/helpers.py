# -*- coding: utf-8 -*-
"""
Helpers functions for all scripts
"""
import ast
import pickle
import spacy
import pandas as pd
from spacy.tokens import DocBin

def ent_to_uri(ent: spacy.tokens.Span) -> str:
    """ From entity return DBpedia URI """
    return ent._.dbpedia_raw_result["@URI"]

def get_spacy_docs_from_bytes(pkl_file: str):
    nlp = spacy.blank("en")
    with open(pkl_file, "rb") as openfile:
        bytes_data = pickle.load(openfile)
    doc_bin = DocBin().from_bytes(bytes_data)
    return list(doc_bin.get_docs(nlp.vocab))

def format_string_col(content):
    """ if str -> convert to Python structure, like dict or list """
    try:
        return ast.literal_eval(content)
    except Exception as _:
        return content

def read_csv(path):
    """ Read csv and convert columns to right format """
    df_ = pd.read_csv(path)
    df_ = df_[[c for c in df_.columns if not c.startswith("Unnamed: ")]]
    for col in ["propbank_output", "sentiment", "polarity_subjectivity", "frame_exist"]:
        if col in df_.columns:
            df_[col] = df_[col].apply(format_string_col)
    return df_
