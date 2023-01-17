# -*- coding: utf-8 -*-
"""
Small demo app
"""
import pickle
import streamlit as st

import spacy
from spacy.tokens import DocBin

def get_data(path="../sample-data/docs_spacy.pkl"):
    """ Loading spacy data from pickled file """
    with open(path, "rb") as openfile:
        bytes_data = pickle.load(openfile)
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_bytes(bytes_data)
    return list(doc_bin.get_docs(nlp.vocab))

def get_entities(doc):
    """ Getting dbpedia entities from spacy doc """
    res = [ent for ent in DOCS[0].ents if ent._.dbpedia_raw_result]
    return list(set([ent._.dbpedia_raw_result["@URI"] for ent in res]))

DOCS = get_data()

st.title("Narratives from tweets + KG")

# Container for text + entities selection
with st.container():
    st.markdown("Choosing text and entities")
    option_order = st.selectbox(
        'Which data do you want to select first',
        ('Text', 'DBpedia entities'))


    if option_order == "Text":
        doc = st.selectbox(
            "Choose a tweet",
            options=DOCS
        )
        st.write(get_entities(doc))
