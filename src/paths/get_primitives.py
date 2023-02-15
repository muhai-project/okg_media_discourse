""" Primitives of nodes/predicates to calculate path scores """
# -*- coding: utf-8 -*-
from collections import defaultdict
import pandas as pd
from rdflib.namespace import RDF, DC, FOAF, SKOS, OWL
from src.graphdb import QUERY_INDEGREE, QUERY_OUTDEGREE, HEADERS_SELECT, main_graphdb

def get_namespace(pred: str):
    """ Extracting ns from predicate """
    return pred.split("#")[0] if "#" in pred else "/".join(pred.split("/")[:-1])

def get_pred_freq(triples: pd.DataFrame, pred: str):
    """ incoming/outgoing predicate frequency as described in RECAP 
    - triples: df with incoming or outgoing nodes
    - pred: predicate of interest"""
    return triples[triples.p == pred].shape[0] / triples.shape[0]

def main_primitives(node):
    """ Calculate primitives, first step to rank the paths between entities """
    ingoing = main_graphdb(
        query=QUERY_INDEGREE.replace("pred_uri", node),
        headers=HEADERS_SELECT)
    outgoing = main_graphdb(
        query=QUERY_OUTDEGREE.replace("pred_uri", node),
        headers=HEADERS_SELECT)

    res_node = {"in": ingoing.shape[0], "ou": outgoing.shape[0]}
    res_node["degree"] = res_node["in"] + res_node["ou"]

    # NODE LEVEL --> From GP paper
    # Namespace variety
    res_node["ns"] = len({get_namespace(pred) for pred in \
        list(ingoing.p.unique()) + list(outgoing.p.unique())})

    # Type degree
    res_node["td"] = outgoing[outgoing.p == f"{str(RDF)}type"] \
        .o.unique().shape[0]

    # Topic outdegree and indegree
    topics = [
        f"{str(DC)}subject", f"{str(FOAF)}primaryTopic",
        f"{str(SKOS)}broader"]
    res_node["so"] = outgoing[outgoing.p.isin(topics)].o.unique().shape[0]
    res_node["si"] = ingoing[ingoing.p.isin(topics)].s.unique().shape[0]

    # Node equality
    node_eq = [f"{str(OWL)}sameAs", f"{str(SKOS)}exactMatch", f"{str(RDF)}seeAlso"]
    res_node["sa"] = len({node for node in \
        list(ingoing[ingoing.p.isin(node_eq)].s.unique()) + \
            list(outgoing[outgoing.p.isin(node_eq)].o.unique())})

    # PRED LEVEL --> From RECAP
    for pred in ingoing.p.unique():
        if "pfi" not in res_node:
            res_node["pfi"] = {}
        res_node["pfi"][pred] = get_pred_freq(triples=ingoing, pred=pred)
    
    for pred in outgoing.p.unique():
        if "pfo" not in res_node:
            res_node["pfo"] = {}
        res_node["pfo"][pred] = get_pred_freq(triples=outgoing, pred=pred)

    return res_node


if __name__ == '__main__':
    NODE = "http://dbpedia.org/resource/French_Revolution"
    main_primitives(node=NODE)
