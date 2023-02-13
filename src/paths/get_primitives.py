# -*- coding: utf-8 -*-

from rdflib.namespace import RDF, DC, FOAF, SKOS, OWL
from src.graphdb import QUERY_INDEGREE, QUERY_OUTDEGREE, main_graphdb

def get_namespace(pred):
    """ Extracting ns from predicate """
    return pred.split("#")[0] if "#" in pred else "/".join(pred.split("/")[:-1])

def main_primitives(node):
    """ Calculate primitives, first step to rank the paths between entities """
    ingoing = main_graphdb(query=QUERY_INDEGREE.replace("pred_uri", node))
    outgoing = main_graphdb(query=QUERY_OUTDEGREE.replace("pred_uri", node))

    res = {"indegree": ingoing.shape[0], "outdegree": outgoing.shape[0]}
    res["degree"] = res["indegree"] + res["outdegree"]

    # Namespace variety
    res["ns"] = len({get_namespace(pred) for pred in \
        list(ingoing.p.unique()) + list(outgoing.p.unique())})

    # Type degree
    res["td"] = outgoing[outgoing.p == f"{str(RDF)}type"] \
        .o.unique().shape[0]

    # Topic outdegree and indegree
    topics = [
        f"{str(DC)}subject", f"{str(FOAF)}primaryTopic",
        f"{str(SKOS)}broader"]
    res["so"] = outgoing[outgoing.p.isin(topics)].o.unique().shape[0]
    res["si"] = ingoing[ingoing.p.isin(topics)].s.unique().shape[0]

    # Node equality
    node_eq = [f"{str(OWL)}sameAs", f"{str(SKOS)}exactMatch", f"{str(RDF)}seeAlso"]
    res["sa"] = len({node for node in \
        list(ingoing[ingoing.p.isin(node_eq)].s.unique()) + \
            list(outgoing[outgoing.p.isin(node_eq)].o.unique())})


NODE = "http://dbpedia.org/resource/French_Revolution"
main_primitives(node=NODE)
