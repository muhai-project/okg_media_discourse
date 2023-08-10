# -*- coding: utf-8 -*-
""" 
Input = tweets with their text + features (pb rolesets, sentiment)
Output = KG
"""
import os
import argparse
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import dask.dataframe as dd
from rdflib import Namespace, Graph, Literal, XSD
from rdflib.namespace import RDF

from src.helpers import format_string_col
from src.logger import Logger

class RDFLIBConverterFromPB:
    """ Converting csv-stored info to rdflib triples
    PB: Propbank 
    RS: RoleSet """

    def __init__(self):
        self.rdf = RDF
        self.observatory = Namespace("http://example.org/muhai/observatory#")
        self.example = Namespace("http://example.com/")
        self.wsj = Namespace("https://w3id.org/framester/wsj/")
        self.pbdata = Namespace("https://w3id.org/framester/pb/pbdata/")
        self.pbschema = Namespace("https://w3id.org/framester/pb/pbschema/")
        self.earmark = Namespace("http://www.essepuntato.it/2008/12/earmark#")
        self.nif = Namespace("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#")
        self.schema = Namespace("http://schema.org/")
        self.xsd = XSD
        self.sioc = Namespace("http://rdfs.org/sioc/ns#")

    def _bind_namespaces(self, graph):
        graph.bind("observatory", self.observatory)
        graph.bind("ex", self.example)
        graph.bind("wsj", self.wsj)
        graph.bind("pbdata", self.pbdata)
        graph.bind("pbschema", self.pbschema)
        graph.bind("earmark", self.earmark)
        graph.bind("nif", self.nif)
        graph.bind("schema", self.schema)
        graph.bind("xsd", self.xsd)
        graph.bind("sioc", self.sioc)
        return graph

    def add_metrics(self, row, post_node, graph):
        """ Adding post metrics """
        sentiment = row.sentiment[0]
        graph.add((post_node, self.observatory["sentiment_label"], Literal(sentiment["label"])))
        graph.add((post_node, self.observatory["sentiment_score"], Literal(sentiment["score"], datatype=self.xsd["float"])))

        pol_subj = row.polarity_subjectivity
        graph.add((post_node, self.observatory["polarity_score"], Literal(pol_subj["polarity"], datatype=self.xsd["float"])))
        graph.add((post_node, self.observatory["subjectivity_score"], Literal(pol_subj["subjectivity"], datatype=self.xsd["float"])))

        return graph

    def process_one_row(self, row, graph):
        """ Add all info for one row """

        # Tweets and sentence
        tweet_id = row.subject.split("_")[1]
        sent_node = self.example[f"{tweet_id}_{row.sentence_id}"]
        post_node = self.example[row.subject]

        graph.add((post_node, self.nif["sentence"], sent_node))
        graph.add((sent_node, self.rdf["value"], Literal(row.sentence_utf8)))
        graph.add((post_node, self.rdf["type"], self.sioc["Post"]))
        graph.add((sent_node, self.rdf["type"], self.nif["Sentence"]))

        # Metrics
        graph = self.add_metrics(row=row, post_node=post_node, graph=graph)

        graph = self.add_rolesets(row=row, graph=graph)
        return graph

    def add_one_roleset(self, data, graph):
        """ Add Corpus Entry for one roleset"""
        tweet_id = data["id"].split("_")[1]
        frame_name = data["info"]["frameName"]
        rs_node = self.example[f"RS_{tweet_id}_{data['i_sent']}_{data['i_frame']}"]
        sent_node = self.example[f"{tweet_id}_{data['i_sent']}"]

        # RS entry
        graph.add((rs_node, self.rdf["type"], self.wsj["CorpusEntry"]))
        # Mapping to RS in Framester
        if data["frame_exist"]:
            rs_framester_node = self.pbdata[frame_name]
        else:
            rs_framester_node = self.observatory[f"roleset/{frame_name}"]
        graph.add((rs_node, self.wsj["onRoleSet"], rs_framester_node))

        # Adding roles
        for i, role_info in enumerate(data["info"]["roles"]):
            role_node = self.example[f"RS_role_{tweet_id}_{data['i_sent']}_{data['i_frame']}_{i}"]
            if role_info["role"] == "V":  # Lexical Unit  
                pred_rs = self.observatory["onToken"]
            else:  # Frame Element
                pred_rs = self.wsj["withmappedrole"]

            if len(role_info["indices"]) == 1:  # nif:Word
                string_type = self.nif["Word"]
                pred_t = self.nif["word"]
                graph.add((role_node, self.observatory["hasTokenIndex"], Literal(int(role_info["indices"][0]), datatype=self.xsd["integer"])))
            else:  # nif:Phrase
                string_type = self.nif["Phrase"]
                pred_t = self.schema["mentions"]
                graph.add((role_node, self.earmark["begins"], Literal(int(role_info["indices"][0]), datatype=self.xsd["integer"])))
                graph.add((role_node, self.earmark["ends"], Literal(int(role_info["indices"][-1]), datatype=self.xsd["integer"])))

            graph.add((role_node, self.rdf["type"], string_type))
            graph.add((role_node, self.rdf["type"], self.wsj["MappedRole"]))
            graph.add((role_node, self.rdf["value"], Literal(role_info["string"])))
            graph.add((rs_node, pred_rs, role_node))

            if role_info["role"] != "V":  # Frame Element
                graph.add((role_node, self.wsj["withpbarg"], self.pbschema[role_info["role"].upper()]))
            
            # Sent to nif:Word and nif:Phrase
            graph.add((sent_node, pred_t, role_node))

        return graph

    def add_rolesets(self, row, graph):
        """ Adding rolesets from PB """
        if isinstance(row.propbank_output, dict):
            frame_set = row.propbank_output['frameSet']
            if isinstance(frame_set, list):
                for i, info in enumerate(frame_set):
                    graph = self.add_one_roleset(
                        data={"id": row.subject, "info": info, "frame_exist": row.frame_exist[i], "i_frame": i, "sentence": row.sentence_utf8, "i_sent": row.sentence_id}, graph=graph)
        return graph

    def __call__(self, input_df):
        graph = Graph()
        graph = self._bind_namespaces(graph=graph)
        for _, row in tqdm(input_df.iterrows(),
                           total=input_df.shape[0]):
            graph = self.process_one_row(row=row, graph=graph)
        return graph


def format_csv(input_df: pd.DataFrame) -> pd.DataFrame:
    """ Transform some string columns into right format (dict, col) """
    for col in ["propbank_output", "sentiment", "polarity_subjectivity", "frame_exist"]:
        input_df[col] = input_df[col].apply(format_string_col)
    return input_df

def convert_df(triples_df):
    """ Function that is parallelized """
    converter = RDFLIBConverterFromPB()
    input_df = format_csv(triples_df.compute())
    return converter(input_df)


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
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', "--path", required=False,
                    help=".csv file with all info from PB")
    ap.add_argument('-f', "--folder", required=False,
                    help="folder with .csv file with all info from PB")
    ap.add_argument('-o', "--output", required=True,
                    help="folder_output")
    args_main = vars(ap.parse_args())

    if not (args_main["path"] or args_main["folder"]):
        raise ValueError("Cannot process further, either `path` arg" + \
            " or `folder` arg must be non empty")

    LOGGER = Logger()
    LOGGER.log_start(name="Building KG from linguistic information")

    if args_main["folder"]:
        DFS = os.listdir(args_main["folder"])
        DFS = [dd.read_csv(os.path.join(args_main["folder"], x )) for x in DFS]
    else:  # args_main["path"]
        DFS = [dd.read_csv(args_main["path"])]

    COUNTER = 0
    for index, DF_TRIPLE in enumerate(DFS):
        print(f"Processing df {index}/{len(DFS)}")
        ARGS = [DF_TRIPLE.get_partition(i) for i in range(DF_TRIPLE.npartitions)]

        RES = main(ARGS)
        for graph_main in RES:
            graph_main.serialize(destination=f"{args_main['output']}/{COUNTER}.ttl",
                                 format='turtle')
            COUNTER += 1

    LOGGER.log_end()
