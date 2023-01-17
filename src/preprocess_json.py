# -*- coding: utf-8 -*-
"""
Preprocessing raw data (.json data directly stored from Twitter API calls)
"""
import os
import re
import json
import pickle
from tqdm import tqdm

import spacy
from spacy.tokens import DocBin
import pandas as pd

PATH = "/Users/ines/Projects/narratives_from_tweets/inequality/data_1212161584904253441.json"
NLP = spacy.load("en_core_web_sm")
NLP.add_pipe("dbpedia_spotlight", config={'confidence': 0.5})

def pre_process_row(row: pd.core.series.Series):
    """ Pre-processing row of df, and esp. the text """
    for old, new in [('#', ' '), ('\n', '. '), (': ', '. '), (' .', '.'), ('’', "'"),
                    ('&amp;', 'and'), ('&lt;', 'less than'), ('&gt;', 'greater than'),
                    ('&le;', 'less-or-equal than'), ('&ge;', 'greater-or-equal than')]:
        row["text"] = row["text"].replace(old, new)
    row["text"] = re.sub('http[s]?:?\/?\/?t?\.?c?o?\/?[A-Za-z0-9\.\/]{0,10}', ' ', row["text"])

    return row

def pre_process_main(folder: str, nlp: spacy.lang.en.English):
    """ Whole pre-processing of one data file (one .json file) """
    files_names = [x for x in os.listdir(folder) if x.startswith("data")][:5]
    data = []

    # Loading .json, converting to pd df with pre-procesed text
    for i in tqdm(range(len(files_names))):
        file_name = files_names[i]
        with open(os.path.join(folder, file_name), "r", encoding="utf-8") as openfile:
            data += json.load(openfile)

    df_data = pd.DataFrame.from_dict(data)[["id", "text"]]

    tqdm.pandas()
    df_data = df_data.progress_apply(pre_process_row, axis=1)
    df_data = df_data.drop_duplicates("text")

    # Spacy + DBpedia Spotlight
    docs = nlp.pipe(df_data.text.values)
    docs = list(docs)

    # Converting to bytes to store as .pkl file
    doc_bin = DocBin(store_user_data=True)
    for doc in docs:
        doc_bin.add(doc)
    bytes_data = doc_bin.to_bytes()

    with open("docs_spacy.pkl", "wb") as openfile:
        pickle.dump(bytes_data, openfile)

pre_process_main(folder="inequality", nlp=NLP)
