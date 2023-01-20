# -*- coding: utf-8 -*-
"""
Helpers functions for all scripts
"""
import pickle
import spacy
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
