# -*- coding: utf-8 -*-
"""
Getting shortest path using GraphDB
Documentation for query: https://graphdb.ontotext.com/documentation/10.1/graph-path-search.html
"""
import os
import io
import argparse
from datetime import datetime
from tqdm import tqdm

import requests
import numpy as np
import pandas as pd

ENDPOINT = "http://localhost:7200/repositories/dbpedia-subset"
HEADERS = {"Accept": "text/csv"}

QUERY_SHORTEST_PATH = """
PREFIX path: <http://www.ontotext.com/path#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX gold: <http://purl.org/linguistics/gold/>
PREFIX geo: <http://www.geonames.org/ontology#>

SELECT ?path ?index_path ?start ?pred ?end 
WHERE {
    VALUES (?src ?dst) {
        ( <src_uri> <dst_uri> )
    }
    SERVICE <http://www.ontotext.com/path#search> {
        <urn:path> path:findPath path:shortestPath ;
                   path:sourceNode ?src ;
                   path:destinationNode ?dst ;
                   path:startNode ?start;
                   path:endNode ?end;
                   path:resultBindingIndex ?index_path ;
                   path:pathIndex ?path ;
                   path:bidirectional true ;
                   #path:poolSize 8;
                   #path:minPathLength -1;
                   #path:maxPathLength 3;
        		   path:exportBinding ?pred .
        SERVICE <urn:path> {
            FILTER (?pred NOT IN (rdfs:seeAlso, rdf:type, dbo:wikiPageWikiLink, dbo:wikiPageRedirects, dbo:wikiPageDisambiguates, foaf:depiction, foaf:isPrimaryTopicOf, dbo:thumbnail, dbo:wikiPageExternalLink, dbo:wikiPageID, dbo:wikiPageLength, dbo:wikiPageRevisionID, dbp:wikiPageUsesTemplate, owl:sameAs, prov:wasDerivedFrom, dbo:wikiPageWikiLinkText, dbo:abstract, rdf:comment, rdf:label, gold:hypernym, dbo:code, geo:featureCode, dbo:simcCode))
            FILTER(strStarts(str(?pred), "http://dbpedia.org"))
            FILTER (isURI(?start)) 
            FILTER (isURI(?end)) 
            ?start ?pred ?end .
        }
    }
}
"""

def get_shortest_path(src_uri, dst_uri, co_occur, query_template=QUERY_SHORTEST_PATH):
    """ GraphDB API call to get shortest paths between entities """
    query = query_template \
        .replace("src_uri", src_uri) \
            .replace("dst_uri", dst_uri)
    start = datetime.now()
    response = requests.get(ENDPOINT, headers=HEADERS,
                            params={"query": query}, timeout=3600)
    end = datetime.now()

    if response.status_code == 200:
        df_pd = pd.read_csv(
            io.StringIO(response.content.decode('utf-8')))

        if df_pd.shape[0] > 0:
            data = {"ent_1": src_uri, "ent_2": dst_uri, "co_occurrence": co_occur,
                    "shortest_path_length": np.max(df_pd.index_path.values) + 1,
                    "nb_paths": np.max(df_pd.path.values) + 1, "time": str(end - start),
                    "error_message": ""}
        else:
            data = {"ent_1": src_uri, "ent_2": dst_uri, "co_occurrence": co_occur,
                    "shortest_path_length": None,
                    "nb_paths": None, "time": str(end - start),
                    "error_message": response.text}
    else:  # requests failed
        df_pd = None
        data = {"ent_1": src_uri, "ent_2": dst_uri, "co_occurrence": co_occur,
                "shortest_path_length": None,
                "nb_paths": None, "time": str(end - start), "error_message": response.text}

    return df_pd, data

# SRC_URI = "http://dbpedia.org/resource/Despotism"
# DST_URI = "http://dbpedia.org/resource/Vladimir_Putin"
# CO_OCCUR = 2111

# DF_PD, DATA = get_shortest_path(SRC_URI, DST_URI, CO_OCCUR)
# DF_PD.to_csv("test_graphdb.csv")
# print(DATA)

if __name__ == '__main__':
    # Example of line in .txt file:
    # http://dbpedia.org/resource/Ukraine	http://dbpedia.org/resource/Vladimir_Putin	2111.0
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', "--entities", required=True,
                    help=".txt file containing, for each line," + \
                        " two entities and one co-occurrence score. Separated by \t")
    ap.add_argument('-o', "--output", required=True,
                    help=".csv output file with main results")
    ap.add_argument('-f', '--folder', default=None,
                    help="folder to save paths if necessary")
    args_main = vars(ap.parse_args())

    folder = args_main["folder"]
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    lines = [x.replace("\n", "").split("\t") \
        for x in open(args_main["entities"], encoding="utf-8").readlines()]

    col_df = ["ent_1", "ent_2", "co_occurrence", "shortest_path_length",
              "nb_paths", "time", "error_message"]
    df_res = pd.DataFrame(columns=col_df)

    for i in tqdm(range(len(lines))):
        [src_uri_, dst_uri_, co_occur_] = lines[i]
        paths, curr_data = get_shortest_path(src_uri=src_uri_, dst_uri=dst_uri_, co_occur=co_occur_)
        df_res = pd.concat([df_res, pd.Series(curr_data).to_frame().T], ignore_index=True)
        df_res.to_csv(args_main["output"])

        if paths is not None and folder:
            src = src_uri_.replace("http://dbpedia.org/resource/", "")
            dst = dst_uri_.replace("http://dbpedia.org/resource/", "")
            paths.to_csv(f"{folder}/{src}_{dst}.csv")
