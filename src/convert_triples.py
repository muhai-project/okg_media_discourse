# -*- coding: utf-8 -*-
""" Converting Python triples/key-values to rdflib triples """
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
        self.earmark = Namespace("http://www.essepuntato.it/2008/12/earmark")

        self.converter = {
            "description": lambda graph, **data: \
                self._convert_sub_pred_literal(self.sioc["content"], graph, **data),
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
                    self.earmark["begins"], XSD.integer, graph, **data),
            "ends": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.earmark["ends"], XSD.integer, graph, **data),
            "hasMatchedNP": lambda graph, **data: \
                self._convert_sub_pred_obj(self.example["hasMatchedNP"], graph, **data),
            "rdf_type": self._add_rdf_type,
            "id": lambda graph, **data: \
                self._convert_sub_pred_literal(self.sioc["id"], graph, **data),
            "createdAt": lambda graph, **data: \
                self._convert_sub_pred_literal_with_type(
                    self.dc_ns["created"], XSD.dateTime, graph, **data),
            "ex_repliedTo": lambda graph, **data: \
                self._convert_reverse_sub_pred_obj(self.sioc["has_reply"], graph, **data),
            "belongsTo": lambda graph, **data: \
                self._convert_sub_pred_obj(self.sioc["has_container"], graph, **data),
            "hasSentence": lambda graph, **data: \
                self._convert_sub_pred_obj(self.example["hasSentence"], graph, **data),
            "refersTo": lambda graph, **data: \
                self._convert_sub_pred_obj(self.example["refersTo"], graph, **data),
        }

        self.rdf_types = {
            "token": self.example["Token"],
            "entity": self.example["Entity"],
            "entity_mention": self.example["EntityMention"],
            "user": self.sioc["User"],
            "tweet": self.sioc["Post"],
            "conversation": self.sioc["Forum"],
            "sentence": self.example["Sentence"]
        }

    def _add_rdf_type(self, graph, **data):
        graph.add((
            self.example[quote(data["subject"])], self.rdf["type"], self.rdf_types[data["object"]]
        ))

        if data["object"] == "entity_mention":
            # adding beginning and ending indexes for the entity mention
            [start, end] = data["subject"].split("#")[-1].split(",")
            graph.add((
            self.example[quote(data["subject"])],
            self.earmark["begins"],
            Literal(start, datatype=XSD.integer)
            ))
            graph.add((
            self.example[quote(data["subject"])], self.earmark["ends"], Literal(end, datatype=XSD.integer)
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
            self.example[quote(data["subject"])], pred, Literal(data["object"], datatype=pred_type)))
        return graph

    def convert_dep_rel(self, graph, **data):
        """ Convert triples with dependency relation information """
        predicate = data["predicate"].split("_")[-1]
        graph.add((self.example[quote(data["subject"])],
                   self.example[predicate],
                   self.example[quote(data["object"])]))
        return graph

    def _bind_namespaces(self, graph):
        graph.bind("sioc", self.sioc)
        graph.bind("nee", self.nee)
        graph.bind("schema", self.schema)
        graph.bind("example", self.example)
        graph.bind("dc", self.dc_ns)
        graph.bind("earmark", self.earmark)
        return graph

    def __call__(self, triples_df):
        graph = Graph()
        graph = self._bind_namespaces(graph=graph)
        for _, row in tqdm(triples_df.iterrows(),
                           total=triples_df.shape[0]):
            data = {"subject": row.subject, "predicate": row.predicate, "object": row.object}
            if row.predicate.startswith("dep_"):  # TOCHECK
                graph = self.convert_dep_rel(graph=graph, **data)
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
    ap.add_argument('-p', "--path", required=True,
                    help=".csv file with triple structure")
    ap.add_argument('-o', "--output", required=True,
                    help="folder_output")
    # ap.add_argument('-c', "--chunk_size", required=True,
    #                 help="chunk size for multiprocessing")
    args_main = vars(ap.parse_args())

    start = datetime.now()
    print(f"Started at {start}")

    # CHUNK_SIZE = int(args_main["chunk_size"])
    # DF_TRIPLE = pd.read_csv(args_main["path"])
    DF_TRIPLE = dd.read_csv(args_main["path"])
    DF_TRIPLE.columns = ["subject", "predicate", "object"]
    # ARGS = chunk_df(df_triples=DF_TRIPLE, chunk_size=CHUNK_SIZE)[-2:]
    ARGS = [DF_TRIPLE.get_partition(i) for i in range(DF_TRIPLE.npartitions)]

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

    RES = main(ARGS)

    for i, graph_main in enumerate(RES):
        graph_main.serialize(destination=f"{args_main['output']}/{i}.ttl", format='turtle')
    
    end = datetime.now()
    print(f"Ended at {end}\nTook {end-start}")
