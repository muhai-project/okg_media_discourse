# -*- coding: utf-8 -*-
""" Converting Python triples/key-values to rdflib triples """
import os
import argparse
from datetime import datetime
import multiprocessing as mp
from urllib.parse import quote
from tqdm import tqdm

# import pandas as pd
import dask.dataframe as dd
from rdflib import URIRef, Namespace, Literal, Graph, XSD
from rdflib.namespace import RDFS, RDF, SKOS

class RDFLIBConverter:
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
        self.observatory = Namespace("http://example.org/muhai/observatory#")
        self.xsd = XSD

        self.converter = {
            "begins": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.earmark["begins"], self.xsd["integer"], graph, **data),  # OK
            "belongsTo": lambda graph, **data: \
                self._convert_sub_pred_obj(self.sioc["has_container"], graph, **data),  # OK
            "createdAt": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.dc_ns["created"], self.xsd["dateTime"], graph, **data),  # OK
            "ends": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.earmark["ends"], self.xsd["integer"], graph, **data),  # OK
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
                    self.observatory["nb_like"], self.xsd["integer"], graph, **data),  # OK
            "metric_nb_repost": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.observatory["nb_repost"], self.xsd["integer"], graph, **data),  # OK
            "rdf_type": self._add_rdf_type,  # OK
            "refersTo": lambda graph, **data: \
                self._convert_reverse_sub_pred_obj(self.nif["anchorOf"], graph, **data),  # OK
            "reply_of": lambda graph, **data: \
                self._convert_sub_pred_obj(self.observatory["reply_of"], graph, **data),  # OK
            "repost_of": lambda graph, **data: \
                self._convert_sub_pred_obj(self.observatory["repost_of"], graph, **data),  # OK
            "tokenIndex": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.observatory["hasTokenIndex"], self.xsd["integer"], graph, **data),  # OK
            "tokenLemma": lambda graph, **data: \
                self._convert_sub_pred_literal(self.nif["lemma"], graph, **data),  # OK
            "tokenPos": lambda graph, **data: \
                self._convert_sub_pred_literal(self.nif["posTag"], graph, **data),  # OK
            "token_partOf": lambda graph, **data: \
                self._convert_reverse_sub_pred_obj(self.nif["superString"], graph, **data),  # OK
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

    def extract_types(self):
        """ entity type to URI type """
        return {
            "conversation": self.sioc["Forum"],  # OK
            "entity": self.observatory["Entity"],  # OK
            "phrase": self.nif["Phrase"],  # OK
            "post": self.sioc["Post"],  # OK
            "reply": self.observatory["Reply"],  # OK
            "repost": self.observatory["RePost"],  # OK
            "sentence": self.nif["Sentence"],  # OK
            "token": self.nif["Word"],  # OK
            "tweet": self.sioc["Post"],  # OK
            "user": self.sioc["User"],  # OK
        }

    def _add_rdf_type(self, graph, **data):
        graph.add((
            self.example[quote(data["subject"])], self.rdf["type"], self.rdf_types[data["object"]]
        ))

        if data["object"] == "phrase":
            # adding beginning and ending indexes for the entity mention
            [start, end] = data["subject"].split("#")[-1].split(",")
            graph.add((
            self.example[quote(data["subject"])],
            self.earmark["begins"],
            Literal(start, datatype=self.xsd["integer"])
            ))
            graph.add((
            self.example[quote(data["subject"])], self.earmark["ends"],
            Literal(end, datatype=self.xsd["integer"])
            ))
        return graph

    def _convert_sub_pred_uriref(self, pred, graph, **data):
        graph.add((
            self.example[quote(data["subject"])], pred, URIRef(data["object"])
        ))
        return graph

    def _convert_sub_pred_obj(self, pred, graph, **data):
        graph.add((
            self.example[quote(data["subject"])], pred, self.example[quote(data["object"])]
        ))
        return graph

    def _convert_reverse_sub_pred_obj(self, pred, graph, **data):
        graph.add((
            self.example[quote(data["object"])], pred, self.example[quote(data["subject"])]
        ))
        return graph

    def _convert_sub_pred_literal(self, pred, graph, **data):
        graph.add((
            self.example[quote(data["subject"])], pred, Literal(data["object"])))
        return graph

    def _convert_sub_pred_literal_with_type(self, pred, pred_type, graph, **data):
        graph.add((
            self.example[quote(data["subject"])], pred, Literal(data["object"],
            datatype=pred_type)))
        return graph

    def convert_dep_rel(self, graph, **data):
        """ Convert triples with dependency relation information """
        predicate = data["predicate"].split("_")[-1]
        graph.add((self.example[quote(data["subject"])],
                   self.observatory[f"dep_rel_{predicate}"],
                   self.example[quote(data["object"])]))
        return graph

    def _bind_namespaces(self, graph):
        graph.bind("sioc", self.sioc)  # OK
        graph.bind("nee", self.nee)  # OK
        graph.bind("schema", self.schema)  # OK
        graph.bind("example", self.example)
        graph.bind("dc", self.dc_ns)  # OK
        graph.bind("earmark", self.earmark)  # OK
        graph.bind("nif", self.nif)  # OK
        graph.bind("observatory", self.observatory)  # OK
        # graph.bind("rdfs", self.rdfs)  # OK
        # graph.bind("rdf", self.rdf)  # OK
        # graph.bind("xsd", self.xsd)  # OK
        return graph

    def __call__(self, triples_df):
        graph = Graph()
        graph = self._bind_namespaces(graph=graph)
        for _, row in tqdm(triples_df.iterrows(),
                           total=triples_df.shape[0]):
            data = {"subject": row.subject, "predicate": row.predicate, "object": row.object}
            if row.predicate.startswith("dep_"):  # TOCHECK
                graph = self.convert_dep_rel(graph=graph, **data)
            elif row.predicate == "description":
                pass
            else:
                graph = self.converter[row.predicate](graph, **data)

        return graph

# def chunk_df(df_triples: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
#     """ Dividing df in chunks for multiprocessing """
#     helper = df_triples.shape[0]//chunk_size
#     res = [df_triples[i*chunk_size: (i+1)*chunk_size] for i in range(helper)]
#     res.append(df_triples[chunk_size*helper:])
#     return res


def convert_df(triples_df):
    """ Function that is parallelized """
    converter = RDFLIBConverter()
    return converter(triples_df.compute())


def main(values):
    """ Main function with multiprocessing """
    with mp.Pool(processes=4) as pool:
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
    # ap.add_argument('-c', "--chunk_size", required=True,
    #                 help="chunk size for multiprocessing")
    args_main = vars(ap.parse_args())

    if not (args_main["path"] or args_main["folder"]):
        raise ValueError("Cannot process further, either `path` arg" + \
            " or `folder` arg must be non empty")

    START = datetime.now()
    print(f"Started at {START}")

    if args_main["folder"]:
        DFS = os.listdir(args_main["folder"])
        DFS = [dd.read_csv(os.path.join(args_main["folder"], x )) for x in DFS]
    else:  # args_main["path"]
        DFS = dd.read_csv(args_main["path"])

    COUNTER = 0
    for i, DF_TRIPLE in enumerate(DFS):
        print(f"Processing df {i}/{len(DFS)}")
        DF_TRIPLE.columns = ["subject", "predicate", "object"]
        ARGS = [DF_TRIPLE.get_partition(i) for i in range(DF_TRIPLE.npartitions)]

        RES = main(ARGS)
        for graph_main in RES:
            graph_main.serialize(destination=f"{args_main['output']}/{COUNTER}.ttl",
                                 format='turtle')
            COUNTER += 1

    END = datetime.now()
    print(f"Ended at {END}\nTook {END-START}")

    # CHUNK_SIZE = int(args_main["chunk_size"])
    # ARGS = chunk_df(df_triples=DF_TRIPLE, chunk_size=CHUNK_SIZE)[-2:]
    # ['has_creator', 'description', 'hasToken', 'label', 'tokenIndex', 'tokenPos',
    # 'tokenLemma', 'begins', 'ends', 'mentions', 'token_partOf', 'hasMatchedNP', 'hasMatchedURL']
    # ['dep_npadvmod', 'dep_advmod', 'dep_intj', 'dep_prep', 'dep_det', 'dep_pobj', 'dep_amod',
    # 'dep_nsubj', 'dep_ROOT', 'dep_attr', 'dep_punct', 'dep_cc', 'dep_conj',
    # 'dep_nummod', 'dep_compound', 'dep_expl', 'dep_acomp', 'dep_mark', 'dep_auxpass',
    # 'dep_ccomp', 'dep_prt', 'dep_dobj', 'dep_appos', 'dep_dep', 'dep_csubj', 'dep_poss',
    # 'dep_nsubjpass', 'dep_aux', 'dep_relcl', 'dep_agent', 'dep_pcomp']

    # CONVERTER = RDFLIBConverter()
    # # GRAPH = CONVERTER(triples_df=df_triple)
    # # GRAPH.serialize(destination=args_main["output"], format='turtle')

    # def main(arg):
    #     converter = RDFLIBConverter()
    #     return converter(arg)

    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     res = pool.map(main, ARGS)
