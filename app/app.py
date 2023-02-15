# -*- coding: utf-8 -*-
"""
Small demo app
"""
import os
import pickle
from collections import defaultdict
import streamlit as st
import streamlit.components.v1 as components

import spacy
from spacy.tokens import DocBin
from spacy import displacy
import pandas as pd

from utils.graph_vis import build_complete_network
from utils.read_data import get_source_code
from utils.streamlit_helpers import init_var, on_change_refresh_first_ent, \
    on_change_refresh_second_ent, on_click_refresh_ent
from utils.path_info import extract_path_pattern
from settings import FOLDER_PATH

st.set_page_config(layout="wide")
init_var(var_list=[("ent_1_boolean", False), ("ent_2_boolean", False)])




def get_data(path=os.path.join(FOLDER_PATH, "sample-data/docs_spacy_inequality.pkl")):
    """ Loading spacy data from pickled file """
    with open(path, "rb") as openfile:
        bytes_data = pickle.load(openfile)
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_bytes(bytes_data)
    return list(doc_bin.get_docs(nlp.vocab))

def get_entities(doc):
    """ Getting dbpedia entities from spacy doc """
    res = [ent for ent in doc.ents if ent._.dbpedia_raw_result]
    return list(set([ent._.dbpedia_raw_result["@URI"] for ent in res]))

def get_ent_to_co_occurr(edges):
    """ For each entity, get related entities (co-occurring in tweets) """
    res = defaultdict(list)
    for [curr_ent_1, curr_ent_2] in edges:
        res[curr_ent_1].append(curr_ent_2)
        res[curr_ent_2].append(curr_ent_1)
    return res

# DOCS = get_data()
# ENTITIES_IN_DOC = [get_entities(doc) for doc in DOCS]

# lines = [x.replace("\n", "").split("\t")[:2] \
#         for x in open(os.path.join(FOLDER_PATH, "sample-data/edges_inequality.txt"), encoding="utf-8").readlines()]
# ENTITIES_TO_CO_OCCUR = get_ent_to_co_occurr(edges=lines)
# ENTITIES = list(ENTITIES_TO_CO_OCCUR.keys())

st.title("Narratives from tweets + KG")

# Container for text + entities selection
with st.container():
    st.markdown("Choosing dataset")
    option_order = st.selectbox(
        'Select your data',
        ("", 'Conflict Ukraine/Russia', "Inequality"))
    
    data_name = "docs_spacy_inequality.pkl" if option_order == 'Inequality' else "docs_spacy_ukraine_russia.pkl"
    data_path = os.path.join(FOLDER_PATH, "sample-data", data_name)
    DOCS = get_data(path=data_path)
    ENTITIES_IN_DOC = [get_entities(doc) for doc in DOCS]

    edge_path = "edges_inequality.txt" if option_order == "Inequality" else "edges_ukraine_russia.txt"
    lines = [x.replace("\n", "").split("\t")[:2] \
        for x in open(os.path.join(FOLDER_PATH, "sample-data", edge_path), encoding="utf-8").readlines()]
    ENTITIES_TO_CO_OCCUR = get_ent_to_co_occurr(edges=lines)
    ENTITIES = list(ENTITIES_TO_CO_OCCUR.keys())


    if option_order:
        ent_1 = st.selectbox("Choose a first entity", options=[""] + ENTITIES, on_change=on_change_refresh_first_ent)

        if ent_1 and st.session_state.ent_1_boolean:
            st.session_state["ent_1"] = ent_1
            ent_2 = st.selectbox("Choose a second entity", options=[""] + ENTITIES_TO_CO_OCCUR[ent_1], on_change=on_change_refresh_second_ent)

            if ent_2 and st.session_state.ent_2_boolean:
                st.session_state["ent_2"] = ent_2
                indexes = [i for i, entities in enumerate(ENTITIES_IN_DOC) \
                    if st.session_state["ent_1"] in entities and \
                        st.session_state["ent_2"] in entities]
                curr_docs = [DOCS[i] for i in indexes]

                with st.expander("Click here to see tweets"):
                    for doc in curr_docs:
                        st.write(doc.text)
                
                [start, end] = sorted([ent_1, ent_2])
                start = start.replace("http://dbpedia.org/resource/", "")
                end = end.replace("http://dbpedia.org/resource/", "")
                path_folder = "paths_inequality" if option_order == 'Inequality' else "paths_ukraine_russia"
                paths_csv = f"{FOLDER_PATH}/{path_folder    }/{start}_{end}.csv"
                paths = pd.read_csv(paths_csv)
                st.write(paths)
                res_path, res_pred = extract_path_pattern(paths=paths)
                st.write(res_path)
                st.write(res_pred)
                if os.path.exists(paths_csv):
                    folder_save = f"./data/{start}_{end}"
                    if not os.path.exists(folder_save):
                        os.makedirs(folder_save)
                    
                    graph_save = os.path.join(folder_save, "connections.html")
                    if not os.path.exists(graph_save):
                        build_complete_network(
                            paths=paths, save_file=graph_save)

                    source_code = get_source_code(html_path=os.path.join(folder_save, "connections.html"))
                    st.write("#")
                    components.html(source_code, width=850, height=850)
        
                button_refresh_ent = st.button("Refresh entities", on_click=on_click_refresh_ent)


    if option_order == "Text":
        st.write('WIP')
        # doc = st.selectbox(
        #     "Choose a tweet",
        #     options=DOCS
        # )

        # st.write(doc)
        # entities = get_entities(doc)
        # st.write(f"The DBpedia entities that were found are: {entities}")
        # ent_1 = st.selectbox(
        #     "Choose a first entity",
        #     options=entities
        # )
        # ent_2 = st.selectbox(
        #     "Choose a second entity",
        #     options=[x for x in entities if x != ent_1]
        # )

        # [start, end] = sorted([ent_1, ent_2])
        # start = start.replace("http://dbpedia.org/resource/", "")
        # end = end.replace("http://dbpedia.org/resource/", "")

        # paths_csv = f"{FOLDER_PATH}/{start}_{end}.csv"
        # if os.path.exists(paths_csv):
        #     folder_save = f"./data/{start}_{end}"
        #     if not os.path.exists(folder_save):
        #         os.makedirs(folder_save)
            
        #     graph_save = os.path.join(
        #                 folder_save, "connections.html")
        #     if not os.path.exists(graph_save):
        #         paths = pd.read_csv(paths_csv)
        #         st.write(paths)
        #         build_complete_network(
        #             paths=paths, save_file=graph_save)

        #     source_code = get_source_code(html_path=os.path.join(folder_save, "connections.html"))
        #     components.html(source_code, width=750, height=750)
