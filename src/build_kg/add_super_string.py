# -*- coding: utf-8 -*-
"""
Adding superstring between tokens and phrases
"""
import os
import argparse
from urllib.parse import quote
from tqdm import tqdm
from rdflib import Graph, Namespace
from src.logger import Logger

EXAMPLE = Namespace("http://example.com/")
NIF = Namespace("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#")


def read_lines(file_path: str) -> list[str]:
    """ Return set of entities from lines of entities """
    with open(file_path, encoding='utf8') as openfile:
        lines = openfile.readlines()
    return list(set(x.replace("\n", "") for x in lines))


def update_info(info: dict, entities: list[str]) -> dict:
    """ info: grouped by id, contains token and phrase index info 
    info = {<id>: {"token": [i1, i2, ..., in], "phrase": [[s1, e1], ..., [sn, en]]}}"""
    for entity in entities:
        id_, nbs = entity.split("_")[1].split("#")

        if id_ not in info:
            info[id_] = {"token": [], "phrase": []}

        if "," in nbs:  # phrase
            start, end = nbs.split(",")
            info[id_]["phrase"].append([int(start), int(end)])
        else:  # token
            info[id_]["token"].append(int(nbs))
    return info


def build_info(file_1: str, file_2: str) -> dict:
    """ Build info from two files """
    ent_1 = read_lines(file_1)
    ent_2 = read_lines(file_2)

    info = {}
    info = update_info(info, entities=ent_1)
    info = update_info(info, entities=ent_2)

    return info


def is_subset(x_1: list[int], x_2: list[int]):
    """ return False if none is a subset of the other, else return
    a, b s.t. contained in b """
    if x_1[0] < x_2[0]:
        return is_subset(x_1=x_2, x_2=x_1)

    if x_1[1] <= x_2[1]:
        return x_1, x_2

    return False


def process_one_sent(graph: Graph, info: dict, id_: int):
    """ Update graph with info on indexes from one sentence"""
    # dealing with tokens: linking tokens to phrases (if any)
    for [start, end] in info['phrase']:
        tokens = [x for x in info['token'] if start <= x <= end]
        for token in tokens:
            graph.add((
                EXAMPLE[quote(f"token_{id_}#{token}")],
                NIF["superString"],
                EXAMPLE[quote(f"ent_{id_}#{start},{end}")]
            ))

    # dealing with phrases: linking phrases to phrases (if any)
    for i, x_1 in enumerate(info['phrase']):
        for _, x_2 in enumerate(info['phrase'][i+1:]):
            subset = is_subset(x_1, x_2)
            if subset:
                [s_1, e_1] = subset[0]
                [s_2, e_2] = subset[1]
                graph.add((
                    EXAMPLE[quote(f"ent_{id_}#{s_1},{e_1}")],
                    NIF["superString"],
                    EXAMPLE[quote(f"ent_{id_}#{s_2},{e_2}")]
                ))

    return graph


def build_superstring_graph(file_1: str, file_2: str) -> Graph:
    """ Build graph based on nif:superString predicate only"""
    info = build_info(file_1, file_2)

    graph = Graph()
    graph.bind("nif", NIF)
    graph.bind("ex", EXAMPLE)
    for id_, info_ in tqdm(info.items()):
        graph = process_one_sent(graph, info_, id_)

    return graph


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f1', "--file1", required=False,
                    help="first file")
    ap.add_argument('-f2', "--file2", required=False,
                    help="second file")
    ap.add_argument('-o', "--output", required=True,
                    help="folder_output")
    args_main = vars(ap.parse_args())

    LOGGER = Logger()
    LOGGER.log_start("Building superstring KG")
    GRAPH = build_superstring_graph(file_1=args_main["file1"], file_2=args_main["file2"])
    GRAPH.serialize(os.path.join(args_main["output"], "superstring.ttl"))
    LOGGER.log_end()
