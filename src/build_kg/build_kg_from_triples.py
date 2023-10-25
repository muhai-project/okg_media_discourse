# -*- coding: utf-8 -*-
""" Converting Python triples/key-values to rdflib triples """
import os
import argparse
import multiprocessing as mp
from urllib.parse import quote
import psutil
from tqdm import tqdm

import pandas as pd
import dask.dataframe as dd
from rdflib import URIRef, Namespace, Literal, Graph, XSD
from rdflib.namespace import RDFS, RDF, SKOS

from src.logger import Logger
from src.helpers import check_args, get_dask_df

class RDFLIBConverterFromTriples:
    """Converting Python triples/key-values to rdflib triples"""

    def __init__(self):
        self.rdfs = RDFS
        self.sioc = Namespace("http://rdfs.org/sioc/ns#")
        self.nee = Namespace("http://www.ics.forth.gr/isl/oae/core#")
        self.rdf = RDF
        self.schema = Namespace("http://schema.org/")
        self.example = Namespace("http://example.com/")
        self.skos = SKOS
        self.dc_ns = Namespace("http://purl.org/dc/terms/")
        self.earmark = Namespace("http://www.essepuntato.it/2008/12/earmark#")
        self.nif = Namespace("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#")
        self.obio = Namespace("https://w3id.org/okg/obio-ontology/")
        self.xsd = XSD

        self.converter = {
            "anchorOf": lambda graph, **data: \
                self._convert_sub_pred_obj(self.nif["anchorOf"], graph, **data),  # OK
            "begins": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.earmark["begins"], self.xsd["int"], graph, **data),  # OK
            "belongsTo": lambda graph, **data: \
                self._convert_sub_pred_obj(self.sioc["has_container"], graph, **data),  # OK
            "createdAt": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.dc_ns["created"], self.xsd["dateTime"], graph, **data),  # OK
            "ends": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.earmark["ends"], self.xsd["int"], graph, **data),  # OK
            "hasMatchedURL": lambda graph, **data: \
                self._convert_sub_pred_uriref(self.nee["hasMatchedURL"], graph, **data),  # OK
            "hasSentence": lambda graph, **data: \
                self._convert_sub_pred_obj(self.nif["sentence"], graph, **data),  # OK
            "hasToken": lambda graph, **data: \
                self._convert_sub_pred_obj(self.nif["word"], graph, **data),  # OK
            "has_creator": lambda graph, **data: \
                self._convert_sub_pred_obj(self.sioc["has_creator"], graph, **data),  # OK
            "id": lambda graph, **data: \
                self._convert_sub_pred_literal(self.sioc["id"], graph, **data),  # OK
            "label": lambda graph, **data: \
                self._convert_sub_pred_literal(self.rdf["value"], graph, **data),  # OK
            "mentions": lambda graph, **data: \
                self._convert_sub_pred_obj(self.schema["mentions"], graph, **data),  # OK
            "metric_nb_like": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.obio["nb_like"], self.xsd["int"], graph, **data),  # OK
            "metric_nb_repost": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.obio["nb_repost"], self.xsd["int"], graph, **data),  # OK
            "rdf_type": self._add_rdf_type,  # OK
            "reply_of": lambda graph, **data: \
                self._convert_sub_pred_obj(self.obio["reply_of"], graph, **data),  # OK
            "repost_of": lambda graph, **data: \
                self._convert_sub_pred_obj(self.obio["repost_of"], graph, **data),  # OK
            "superString": lambda graph, **data: \
                self._convert_sub_pred_obj(self.nif["superString"], graph, **data),  # OK
            "tokenIndex": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.obio["hasTokenIndex"], self.xsd["int"], graph, **data),  # OK
            "tokenLemma": lambda graph, **data: \
                self._convert_sub_pred_literal(self.nif["lemma"], graph, **data),  # OK
            "tokenPos": lambda graph, **data: \
                self._convert_sub_pred_literal(self.nif["posTag"], graph, **data),  # OK
            ############################################################
            # TO BE CHECKED
            "is_verified": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.obio["is_verified"], self.xsd["boolean"], graph, **data),  # boolean
            "user_label": lambda graph, **data: \
                self._convert_sub_pred_literal(self.rdfs["label"], graph, **data),  # literal
            "user_description": lambda graph, **data: \
                self._convert_sub_pred_literal(
                    self.obio["description"], graph, **data),   # literal
            "follower": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.obio["follower"], self.xsd["int"], graph, **data),   # int
            "following": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.obio["following"], self.xsd["int"], graph, **data),   # int
            "location": lambda graph, **data: \
                self._convert_sub_pred_literal(
                    self.obio["location"], graph, **data),  # literal

            ############################################################
            # BELOW in previous models, OBSOLETE
            # "description" removed due to Twitter new privacy
            # "description": lambda graph, **data: \
            #     self._convert_sub_pred_literal(self.sioc["content"], graph, **data),
            # "hasMatchedNP": lambda graph, **data: \
            #     self._convert_sub_pred_obj(self.example["hasMatchedNP"], graph, **data),
            # "ex_repliedTo": lambda graph, **data: \
            #     self._convert_reverse_sub_pred_obj(self.sioc["has_reply"], graph, **data),
        }

        self.rdf_types = self.extract_types()
        self.superstring_cand = []

    def extract_types(self):
        """ entity type to URI type """
        return {
            "conversation": self.sioc["Forum"],  # OK
            "entity": self.obio["Entity"],  # OK
            "phrase": self.nif["Phrase"],  # OK
            "post": self.sioc["Post"],  # OK
            "reply": self.obio["Reply"],  # OK
            "repost": self.obio["RePost"],  # OK
            "sentence": self.nif["Sentence"],  # OK
            "token": self.nif["Word"],  # OK
            "tweet": self.sioc["Post"],  # OK
            "user": self.sioc["User"],  # OK
        }

    def _add_rdf_type(self, graph: Graph, **data: dict) -> Graph:
        graph.add((
            self.example[quote(data["subject"])], self.rdf["type"], self.rdf_types[data["object"]]
        ))

        if data["object"] == "phrase":
            # adding beginning and ending indexes for the entity mention
            [start, end] = data["subject"].split("#")[-1].split(",")
            graph.add((self.example[quote(data["subject"])],
                       self.earmark["begins"],
                       Literal(start, datatype=self.xsd["int"])))
            graph.add((self.example[quote(data["subject"])], self.earmark["ends"],
                       Literal(end, datatype=self.xsd["int"])))

        if data["object"] in ["phrase", "token"]:
            self.superstring_cand.append(data["subject"])

        return graph

    def _convert_sub_pred_uriref(self, pred: URIRef, graph: Graph, **data: dict) -> Graph:
        graph.add((
            self.example[quote(data["subject"])], pred, URIRef(data["object"])
        ))
        return graph

    def _convert_sub_pred_obj(self, pred: URIRef, graph: Graph, **data: dict) -> Graph:
        graph.add((
            self.example[quote(data["subject"])], pred, self.example[quote(data["object"])]
        ))
        return graph

    def _convert_reverse_sub_pred_obj(self, pred: URIRef, graph: Graph, **data: dict) -> Graph:
        graph.add((
            self.example[quote(data["object"])], pred, self.example[quote(data["subject"])]
        ))
        return graph

    def _convert_sub_pred_literal(self, pred: URIRef, graph: Graph, **data: dict) -> Graph:
        graph.add((
            self.example[quote(data["subject"])], pred, Literal(data["object"])))
        return graph

    def _convert_sub_pred_literal_with_type(self, pred: URIRef, pred_type: URIRef,
                                            graph: Graph, **data: dict) -> Graph:
        graph.add((
            self.example[quote(data["subject"])], pred, Literal(data["object"],
            datatype=pred_type)))
        return graph

    def convert_dep_rel(self, graph: Graph, **data: dict) -> Graph:
        """ Convert triples with dependency relation information """
        predicate = data["predicate"].split("_")[-1]
        graph.add((self.example[quote(data["subject"])],
                   self.obio[f"dep_rel_{predicate}"],
                   self.example[quote(data["object"])]))
        return graph

    def _bind_namespaces(self, graph: Graph) -> Graph:
        graph.bind("sioc", self.sioc)  # OK
        graph.bind("nee", self.nee)  # OK
        graph.bind("schema", self.schema)  # OK
        graph.bind("example", self.example)
        graph.bind("dc", self.dc_ns)  # OK
        graph.bind("earmark", self.earmark)  # OK
        graph.bind("nif", self.nif)  # OK
        graph.bind("obio", self.obio)  # OK
        # graph.bind("rdfs", self.rdfs)  # OK
        # graph.bind("rdf", self.rdf)  # OK
        # graph.bind("xsd", self.xsd)  # OK
        return graph

    def __call__(self, triples_df: pd.DataFrame) -> (Graph, list[str]):
        graph = Graph()
        graph = self._bind_namespaces(graph=graph)
        for _, row in tqdm(triples_df.iterrows(),
                           total=triples_df.shape[0]):
            data = {"subject": row.subject, "predicate": row.predicate, "object": row.object}
            if row.object:
                if row.predicate.startswith("dep_"):  # TOCHECK
                    graph = self.convert_dep_rel(graph=graph, **data)
                elif row.predicate == "description":
                    pass
                else:
                    graph = self.converter[row.predicate](graph, **data)

        return graph, self.superstring_cand


def convert_df(triples_df: dd.core.DataFrame) -> (Graph, list[str]):
    """ Function that is parallelized """
    converter = RDFLIBConverterFromTriples()
    return converter(triples_df.compute())


def main(values: list) -> list[(Graph, list[str])]:
    """ Main function with multiprocessing """
    with mp.Pool(processes=psutil.cpu_count()) as pool:
        results = []
        for result in tqdm(pool.map(convert_df, values),
                           total=len(values)):
            results.append(result)

        pool.close()
        pool.join()
    return results

if __name__ == '__main__':
    # python src/convert_triples.py -p ./sample-data/2023_05_17_example_triples_big.csv \
    # -o ./sample-data/2023_01_ams/
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', "--path", required=False,
                    help=".csv file with triple structure")
    ap.add_argument('-f', "--folder", required=False,
                    help="folder with .csv file with triple structure")
    ap.add_argument('-o', "--output", required=True,
                    help="folder_output")
    args_main = vars(ap.parse_args())

    check_args(args=args_main)

    DFS = get_dask_df(args=args_main)

    COUNTER = 0

    LOGGER = Logger()
    LOGGER.log_start(name="Building KG from triples")
    for i, (DF_TRIPLE, _) in enumerate(DFS):
        print(f"Processing df {i}/{len(DFS)}")
        DF_TRIPLE = DF_TRIPLE[[col for col in DF_TRIPLE if col != "Unnamed: 0"]]
        DF_TRIPLE.columns = ["subject", "predicate", "object"]
        ARGS = [DF_TRIPLE.get_partition(i) for i in range(DF_TRIPLE.npartitions)]

        RES = main(ARGS)
        CANDS = []
        for graph_main, cands in RES:
            graph_main.serialize(destination=f"{args_main['output']}/{COUNTER}.ttl",
                                 format='turtle')
            COUNTER += 1
            CANDS += cands

    f = open(os.path.join(args_main['output'], "superstring_cands.txt"), "w+", encoding='utf8')
    for cand in CANDS:
        f.write(f"{cand}\n")
    f.close()

    LOGGER.log_end()
