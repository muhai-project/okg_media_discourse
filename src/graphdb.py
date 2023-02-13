# -*- coding: utf-8 -*-
"""
GraphDB related queries
"""
import io
import requests
import argparse
import pandas as pd

ENDPOINT = "http://localhost:7200/repositories/dbpedia-subset"
HEADERS = {"Accept": "text/csv"}


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

def main_graphdb(query):
    """ Curl requests to retrieve info from
    graphdb endpoint + sparql query"""
    response = requests.get(ENDPOINT, headers=HEADERS,
                            params={"query": query}, timeout=3600)
    return pd.read_csv(
        io.StringIO(response.content.decode('utf-8')))

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
    res = main_graphdb(query=QUERY)
    print(res)
