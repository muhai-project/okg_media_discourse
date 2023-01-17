# -*- coding: utf-8 -*-
"""
Getting types of entities that were mentioned in Twitter data
"""
import io
import json
from collections import defaultdict
from tqdm import tqdm

import requests
import pandas as pd

ENDPOINT = "http://localhost:7200/repositories/dbpedia-subset"
HEADERS = {"Accept": "text/csv"}

QUERY_TYPE = """
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?src ?type ?obj
WHERE {
    VALUES (?src ?type) {
        ( <src_uri> rdf:type
        )
    }
    ?src ?type ?obj
    FILTER (regex(str(?obj), "http://dbpedia.org/ontology/[a-zA-Z]"))
    FILTER (!regex(str(?obj), "Animal|Species|Eukaryote|Place|PopulatedPlace|Artist|Settlement|Country|Politician|OfficeHolder|Agent|MilitaryPerson"))
}
"""

def get_type(src_uri, query=QUERY_TYPE):
    """ GraphDB API call to get entity types
    Filtered: only rdf:type and dbo ontology + english """
    query = query \
        .replace("src_uri", src_uri)
    response = requests.get(ENDPOINT, headers=HEADERS,
                            params={"query": query}, timeout=3600)
    if response.status_code == 200:
        return pd.read_csv(
            io.StringIO(response.content.decode('utf-8'))).obj.values
    return []

args_main = {"entities": "edges_ukraine_russia.txt"}
lines = [x.replace("\n", "").split("\t")[:2] \
        for x in open(args_main["entities"], encoding="utf-8").readlines()]
common_paths = defaultdict(int)

for i in tqdm(range(len(lines))):
    [ent_1, ent_2] = lines[i]
    types_1 = get_type(src_uri=ent_1)
    types_2 = get_type(src_uri=ent_2)
    for type_1 in types_1:
        for type_2 in types_2:
            ordered = [x.replace("http://dbpedia.org/ontology/", "") \
                for x in sorted([type_1, type_2])]
            name = f"[{ordered[0]}]_[{ordered[1]}]"
            common_paths[name] += 1

    with open("types.json", "w", encoding='utf-8') as openfile:
        data_json = dict(sorted(common_paths.items(),
                                key=lambda x:x[1], reverse=True))
        json.dump(data_json, openfile, indent=4)
