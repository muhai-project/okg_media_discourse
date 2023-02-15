# -*- coding: utf-8 -*-
"""
GraphDB related queries
"""
import io
import requests
import argparse
import pandas as pd

ENDPOINT = "http://localhost:7200/repositories/dbpedia-subset"
HEADERS_SELECT = {"Accept": "text/csv"}
HEADERS_ASK = {"Accept": "application/json"}


QUERY_PRED_COUNT = """
SELECT (COUNT(?o) as ?count)
WHERE { 
    VALUES (?p) {
        ( <pred_uri> )
    }
	?s ?p ?o .
} GROUPBY ?p
"""

QUERY_INDEGREE = """
SELECT ?s ?p ?o
WHERE { 
    VALUES (?o) {
        ( <pred_uri> )
    }
	?s ?p ?o .
}
"""

QUERY_OUTDEGREE = """
SELECT ?s ?p ?o
WHERE { 
    VALUES (?s) {
        ( <pred_uri> )
    }
	?s ?p ?o .
}
"""

ASK_TRIPLE_IN_KG = """
ASK  { <sub_uri> <pred_uri> <obj_uri> }
"""

def main_graphdb(query, headers, type_o="csv"):
    """ Curl requests to retrieve info from
    graphdb endpoint + sparql query"""
    response = requests.get(ENDPOINT, headers=headers,
                            params={"query": query}, timeout=3600)

    if type_o == "csv":
        return pd.read_csv(
            io.StringIO(response.content.decode('utf-8')))
    if type_o == "json":
        return response.json()
    raise ValueError(f"`type_o` {type_o} not implemented")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-q', "--query_type", required=True,
                    help="type of SPARQL query to run (see keys in dictionary below for options)")
    ap.add_argument('-u', "--uri", required=True,
                    help="URI for templated sparql query")
    args_main = vars(ap.parse_args())

    TYPE_TO_QUERY = {
        "pred_count": QUERY_PRED_COUNT,
        "indegree": QUERY_INDEGREE,
        "outdegree": QUERY_OUTDEGREE
    }

    QUERY = TYPE_TO_QUERY[args_main["query_type"]].replace("pred_uri", args_main["uri"])
    res = main_graphdb(query=QUERY, headers=HEADERS_SELECT)
    print(res)
