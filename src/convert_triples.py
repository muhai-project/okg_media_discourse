# -*- coding: utf-8 -*-
""" Converting Python triples/key-values to rdflib triples """
import argparse
from tqdm import tqdm
import pandas as pd
from rdflib import URIRef, Namespace, Literal, Graph, XSD
from rdflib.namespace import RDFS, RDF, SKOS

class RDLIBConverter:
    """Converting Python triples/key-values to rdflib triples"""
    def __init__(self):
        self.rdfs = RDFS
        self.sioc = Namespace("http://rdfs.org/sioc/ns#")
        self.nee = Namespace("http://www.ics.forth.gr/isl/oae/core#")
        self.rdf = RDF
        self.schema = Namespace("http://schema.org/")
        self.example = Namespace("http://example.com/")
        self.skos = SKOS
        self.dc = Namespace("http://purl.org/dc/terms/")

        self.converter = {
            "description": lambda graph, **data: \
                self._convert_sub_pred_literal(self.example["description"], graph, **data),
            "has_creator": lambda graph, **data: \
                self._convert_sub_pred_obj(self.sioc["has_creator"], graph, **data),
            "hasToken": lambda graph, **data: \
                self._convert_sub_pred_obj(self.example["hasToken"], graph, **data),
            "hasMatchedURL": lambda graph, **data: \
                self._convert_sub_pred_uriref(self.nee["hasMatchedURL"], graph, **data),
            "mentions": lambda graph, **data: \
                self._convert_sub_pred_obj(self.schema["mentions"], graph, **data),
            "token_partOf": lambda graph, **data: \
                self._convert_sub_pred_obj(self.example["partOf"], graph, **data),
            "label": lambda graph, **data: \
                self._convert_sub_pred_literal(self.rdfs["label"], graph, **data),
            "tokenIndex": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.example["hasTokenIndex"], XSD.integer, graph, **data),
            "tokenPos": lambda graph, **data: \
                self._convert_sub_pred_literal(self.example["hasTokenPos"], graph, **data),
            "tokenLemma": lambda graph, **data: \
                self._convert_sub_pred_literal(self.example["hasTokenLemma"], graph, **data),
            "begins": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.example["begins"], XSD.integer, graph, **data),
            "ends": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.example["ends"], XSD.integer, graph, **data),
            "hasMatchedNP": lambda graph, **data: \
                self._convert_sub_pred_obj(self.example["hasMatchedNP"], graph, **data),
            "rdf_type": self._add_rdf_type,
            "id": lambda graph, **data: \
                self._convert_sub_pred_literal(self.sioc["id"], graph, **data),
            "createdAt": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.dc["created"], XSD.dateTime, graph, **data),
            "ex_repliedTo": lambda graph, **data: \
                self._convert_sub_pred_obj(self.example["repliedTo"], graph, **data),
            "belongsTo": lambda graph, **data: \
                self._convert_sub_pred_obj(self.example["belongsTo"], graph, **data), 
        }

        self.rdf_types = {
            "token": self.example["Token"],
            "entity": self.example["Entity"],
            "user": self.sioc["User"],
            "tweet": self.sioc["Post"],
            "conversation": self.example["Conversation"],
        }

    def _add_rdf_type(self, graph, **data):
        graph.add((
            self.example[data["subject"]], self.rdf["type"], self.rdf_types[data["object"]]
        ))
        return graph

    def _convert_sub_pred_uriref(self, pred, graph, **data):
        graph.add((
            self.example[data["subject"]], pred, URIRef(data["object"])
        ))
        return graph
    
    def _convert_sub_pred_obj(self, pred, graph, **data):
        graph.add((
            self.example[data["subject"]], pred, self.example[data["object"]]
        ))
        return graph

    def _convert_sub_pred_literal(self, pred, graph, **data):
        graph.add((
            self.example[data["subject"]], pred, Literal(data["object"])))
        return graph

    def _convert_sub_pred_literal_with_type(self, pred, pred_type, graph, **data):
        graph.add((
            self.example[data["subject"]], pred, Literal(data["object"], datatype=pred_type)))
        return graph

    def convert_dep_rel(self, graph, **data):
        """ Convert triples with dependency relation information """
        predicate = data["predicate"].split("_")[-1]
        graph.add((self.example[data["subject"]], self.example[predicate], self.example[data["object"]]))
        return graph
    
    def _bind_namespaces(self, graph):
        graph.bind("sioc", self.sioc)
        graph.bind("nee", self.nee)
        graph.bind("schema", self.schema)
        graph.bind("example", self.example)
        graph.bind("dc", self.dc)
        return graph

    def __call__(self, triples_df):
        graph = Graph()
        graph = self._bind_namespaces(graph=graph)
        for _, row in tqdm(triples_df.iterrows(), total=triples_df.shape[0]):
            data = {"subject": row.subject, "predicate": row.predicate, "object": row.object}
            if row.predicate.startswith("dep_"):  # TOCHECK
                graph = self.convert_dep_rel(graph=graph, **data)
            else:
                graph = self.converter[row.predicate](graph, **data)

        return graph


if __name__ == '__main__':
    # python src/convert_triples.py -p ./sample-data/2023_01_ams/example_triples.csv -o ./sample-data/2023_01_ams/sample_tweet_kg.ttl
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', "--path", required=True,
                    help=".csv file with triple structure")
    ap.add_argument('-o', "--output", required=True,
                    help=".csv file with triple structure")
    args_main = vars(ap.parse_args())

    df_triple = pd.read_csv(args_main["path"])
    df_triple.columns = ["subject", "predicate", "object"]

    # ['has_creator', 'description', 'hasToken', 'label', 'tokenIndex', 'tokenPos', 'tokenLemma', 'begins', 'ends', 'mentions', 'token_partOf', 'hasMatchedNP', 'hasMatchedURL']
    # ['dep_npadvmod', 'dep_advmod', 'dep_intj', 'dep_prep', 'dep_det', 'dep_pobj', 'dep_amod', 'dep_nsubj', 'dep_ROOT', 'dep_attr', 'dep_punct', 'dep_cc', 'dep_conj', 'dep_nummod', 'dep_compound', 'dep_expl', 'dep_acomp', 'dep_mark', 'dep_auxpass', 'dep_ccomp', 'dep_prt', 'dep_dobj', 'dep_appos', 'dep_dep', 'dep_csubj', 'dep_poss', 'dep_nsubjpass', 'dep_aux', 'dep_relcl', 'dep_agent', 'dep_pcomp']

    CONVERTER = RDLIBConverter()
    GRAPH = CONVERTER(triples_df=df_triple)
    GRAPH.serialize(destination=args_main["output"], format='turtle')
