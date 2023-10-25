# -*- coding: utf-8 -*-
""" 
Input = tweets with their text + features (pb rolesets, sentiment)
Output = KG
"""
import os
import io
import re
import argparse
from urllib.parse import quote
import multiprocessing as mp
import requests
from tqdm import tqdm
import pandas as pd
from rdflib import Namespace, Graph, Literal, XSD, URIRef
from rdflib.namespace import RDF

from src.helpers import read_csv, get_dask_df, check_args
from src.logger import Logger
from settings import SPARQL_ENDPOINT

class RDFLIBConverterFromPB:
    """ Converting csv-stored info to rdflib triples
    PB: Propbank 
    RS: RoleSet """

    def __init__(self):
        self.rdf = RDF
        self.obio = Namespace("https://w3id.org/okg/obio-ontology/")
        self.example = Namespace("http://example.com/")
        self.wsj = Namespace("https://w3id.org/framester/wsj/")
        self.pbdata = Namespace("https://w3id.org/framester/pb/data/")
        self.pbschema = Namespace("https://w3id.org/framester/pb/pbschema/")
        self.earmark = Namespace("http://www.essepuntato.it/2008/12/earmark#")
        self.nif = Namespace("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#")
        self.schema = Namespace("http://schema.org/")
        self.xsd = XSD
        self.sioc = Namespace("http://rdfs.org/sioc/ns#")

        # Unique role sets, to add roles from Framester KG
        self.distinct_pb_role_sets = []
        self.headers = {"Accept": "text/csv"}
        self.sparql_endpoint = SPARQL_ENDPOINT
        self.init_query()

        self.superstring_cand = []

    def init_query(self):
        """ Query to get roles from roleset """
        self.role_query = \
        """
        PREFIX pbschema: <https://w3id.org/framester/pb/pbschema/>
        SELECT DISTINCT ?role
        WHERE {
            <to-change> pbschema:hasRole ?role .
        }   
        """

    def _bind_namespaces(self, graph: Graph) -> Graph:
        """ Binding namespace to graph """
        graph.bind("obio", self.obio)
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

    def add_metrics(self, row: pd.core.series.Series, post_node: URIRef, graph: Graph) -> Graph:
        """ Adding post metrics: sentiment, polarity, subjectivity """
        sentiment = row.sentiment[0]
        graph.add((post_node, self.obio["sentiment_label"], Literal(sentiment["label"])))
        graph.add((post_node, self.obio["sentiment_score"],
                   Literal(sentiment["score"], datatype=self.xsd["float"])))

        pol_subj = row.polarity_subjectivity
        graph.add((post_node, self.obio["polarity_score"],
                   Literal(pol_subj["polarity"], datatype=self.xsd["float"])))
        graph.add((post_node, self.obio["subjectivity_score"],
                   Literal(pol_subj["subjectivity"], datatype=self.xsd["float"])))

        return graph

    def process_one_row(self, row: pd.core.series.Series, graph: Graph,
                        offset: int) -> Graph:
        """ Add all info for one row """

        # Updating offset -> if new tweet, reset to 0
        if row.sent_id == 0:
            offset = 0

        # Tweets and sentence
        tweet_id = row.subject.split("_")[1]
        sent_node = self.example[f"sentence_{tweet_id}#{row.sent_id}"]
        post_node = self.example[row.subject]

        graph.add((post_node, self.nif["sentence"], sent_node))
        # graph.add((sent_node, self.rdf["value"], Literal(row.sent)))
        graph.add((post_node, self.rdf["type"], self.sioc["Post"]))
        graph.add((sent_node, self.rdf["type"], self.nif["Sentence"]))

        # Metrics
        graph = self.add_metrics(row=row, post_node=post_node, graph=graph)

        # Rolesets
        graph = self.add_rolesets(row=row, graph=graph, offset=offset)

        # Updating offset
        offset += row.sent_len

        return graph, offset

    def add_one_roleset(self, data: dict, graph: Graph, offset: int) -> Graph:
        """ Add Corpus Entry for one roleset"""
        tweet_id = data["id"].split("_")[1]
        frame_name = data["info"]["frameName"]
        rs_node = self.example[f"RS_{tweet_id}_{data['i_sent']}_{data['i_frame']}"]
        sent_node = self.example[f"sentence_{tweet_id}#{data['i_sent']}"]

        # RS entry
        graph.add((rs_node, self.rdf["type"], self.wsj["CorpusEntry"]))
        # Mapping to RS in Framester
        if data["frame_exist"]:
            rs_framester_node = self.pbdata[frame_name]
            if rs_framester_node not in self.distinct_pb_role_sets:
                self.distinct_pb_role_sets.append(rs_framester_node)
        else:
            rs_framester_node = self.obio[f"roleset/{frame_name}"]
        graph.add((rs_node, self.wsj["onRoleSet"], rs_framester_node))

        # Adding roles
        for i, role_info in enumerate(data["info"]["roles"]):
            role_node = self.example[f"RS_role_{tweet_id}_{data['i_sent']}_{data['i_frame']}_{i}"]

            if role_info["role"] == "V":  # Lexical Unit
                pred_rs = self.obio["onToken"]
            else:  # Frame Element
                pred_rs = self.wsj["withmappedrole"]

            if len(role_info["indices"]) == 1:  # nif:Word
                string_type = self.nif["Word"]
                pred_t = self.nif["word"]
                try:
                    token_index = offset + data["mapping"][int(role_info["indices"][0])]
                    role_node_id = f"token_{tweet_id}#{token_index}"
                    role_node = self.example[quote(role_node_id)]
                    self.superstring_cand.append(role_node_id)
                    graph.add((role_node, self.obio["hasTokenIndex"],
                               Literal(token_index, datatype=self.xsd["int"])))
                except Exception as exception:  # mapping not found
                    print(exception)
            else:  # nif:Phrase
                string_type = self.nif["Phrase"]
                pred_t = self.nif["subString"]


                try:
                    token_index_s = offset + int(data["mapping"][int(role_info["indices"][0])])
                    token_index_e = offset + int(data["mapping"][int(role_info["indices"][-1])])
                    role_node_id = f"ent_{tweet_id}#{token_index_s},{token_index_e}"
                    role_node = self.example[quote(role_node_id)]
                    self.superstring_cand.append(role_node_id)
                    graph.add((
                        role_node, self.earmark["begins"],
                        Literal(token_index_s, datatype=self.xsd["int"])))
                    graph.add((
                        role_node, self.earmark["ends"],
                        Literal(token_index_e, datatype=self.xsd["int"])))
                except Exception as exception:
                    print(exception)
            
            # adding link between entity and tweet
            graph.add((
                self.example[f"tweet_{tweet_id}"], self.schema["mentions"], role_node
            ))

            graph.add((role_node, self.rdf["type"], string_type))
            graph.add((role_node, self.rdf["type"], self.wsj["MappedRole"]))
            graph.add((role_node, self.rdf["value"], Literal(role_info["string"])))
            graph.add((rs_node, pred_rs, role_node))

            if role_info["role"] != "V":  # Frame Element
                # if matches pattern 'ARG\\d{1}' -> in framester
                role = role_info["role"].upper()
                if re.match("ARG\\d{1}", role):
                    graph.add((role_node, self.wsj["withpbarg"],
                            self.pbschema[role]))
                else:  # other annotations
                    graph.add((role_node, self.wsj["withpbarg"],
                               self.obio[f"pbarg/{role}"]))

            # Sent to nif:Word and nif:Phrase
            graph.add((sent_node, pred_t, role_node))

        return graph

    def add_rolesets(self, row: pd.core.series.Series, graph: Graph, offset: int) -> Graph:
        """ Adding rolesets from PB """
        if isinstance(row.propbank_output, dict) and \
            "frameSet" in row.propbank_output:
            frame_set = row.propbank_output['frameSet']
            # Checking that (1) there are frames (2) there is a token mapping
            if isinstance(frame_set, list) and row.sent_mapping:
                for i, info in enumerate(frame_set):
                    graph = self.add_one_roleset(
                        data={"id": row.subject, "info": info, "frame_exist": row.frame_exist[i],
                              "i_frame": i, "i_sent": row.sent_id, "mapping": row.sent_mapping},
                        graph=graph, offset=offset)
        return graph

    def run_query(self, query: str) -> pd.DataFrame:
        """ Using curl requests to run query """
        response = requests.get(
            self.sparql_endpoint, headers=self.headers,
            params={"query": query}, timeout=3600)
        # print(response.url)
        return pd.read_csv(io.StringIO(response.content.decode('utf-8')))

    def get_roles_framester(self, rs_pb: URIRef) -> list[str]:
        """ SPARQL query to retrieve roles """
        query = self.role_query.replace("to-change", str(rs_pb))
        res = self.run_query(query)
        if res.shape[0] > 0:
            return res.role.values
        return []


    def __call__(self, input_df: pd.DataFrame) -> (Graph, list[str]):
        graph = Graph()
        graph = self._bind_namespaces(graph=graph)

        # Processing each row (= one sentence)
        print(f"Processing {input_df.shape[0]} rows")
        offset = 0
        for _, row in tqdm(input_df.iterrows(),
                           total=input_df.shape[0]):
            graph, offset = self.process_one_row(row=row, graph=graph, offset=offset)

        # Adding role from Propbank frames
        print(f"Processing {len(self.distinct_pb_role_sets)} frame templates")
        for i in tqdm(range(len(self.distinct_pb_role_sets))):
            rs_pb = self.distinct_pb_role_sets[i]
            roles = self.get_roles_framester(rs_pb=rs_pb)
            for role in roles:
                graph.add((rs_pb, self.pbschema["hasRole"], URIRef(role)))

        return graph, self.superstring_cand


def convert_df(triples_df: pd.DataFrame) -> ((Graph, list[str]), str):
    """ Function that is parallelized """
    converter = RDFLIBConverterFromPB()
    input_df = read_csv(triples_df[0].compute())
    return converter(input_df), triples_df[1]


def main(values: list[str]) -> list:
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

    check_args(args=args_main)

    LOGGER = Logger()
    LOGGER.log_start(name="Building KG from linguistic information")

    DFS = get_dask_df(args=args_main)
    if args_main["output"]:
        DFS = [x for x in DFS if not os.path.exists(
            os.path.join(
                args_main["output"],
                f"{x[1].replace('.csv', '').split('_')[1]}.ttl"))]

    CANDS = []
    # Running by batches of 100
    nb_batch = len(DFS)//100 + 1
    for I in range(nb_batch):
        print(f"Running batch {I+1}/{nb_batch} ({round(100*(I+1)/nb_batch, 2)}%)".upper())
        if I == nb_batch:
            CURR_DFS = DFS[(I+1)*100:]
        else:
            CURR_DFS = DFS[I*100:(I+1)*100]

        RESULTS = main(values=CURR_DFS)

        if args_main["output"]:
            for output, index in RESULTS:
                index = index.replace(".csv", "").split("_")[1]
                output[0].serialize(
                    destination=os.path.join(args_main["output"], f"{index}.ttl"),
                    format='turtle')
                CANDS += output[1]

    f = open(os.path.join(args_main['output'], "superstring_cands.txt"), "w+", encoding='utf8')
    for cand in CANDS:
        f.write(f"{cand}\n")
    f.close()

    LOGGER.log_end()
