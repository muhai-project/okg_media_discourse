# -*- coding: utf-8 -*-
""" Visualisation helpers for graph """
from pyvis.network import Network

def pre_process(node):
    """ URI > more human-readable """
    return node.split("/")[-1].replace('_', ' ')

def build_complete_network(paths, save_file):
    """ Build html network after one iteration """
    nt_subgraph = Network("850px", "850px",
                           notebook=False, directed=False)
    nodes = list(set(list(paths.start.values) + list(paths.end.values)))


    for node in nodes:
        nt_subgraph.add_node(pre_process(node), label=pre_process(node))
    for _, row in paths.iterrows():
        nt_subgraph.add_edge(pre_process(row.start),
                             pre_process(row.end),
                             label=pre_process(row.pred))

    nt_subgraph.repulsion(node_distance=600, spring_length=340,
                          spring_strength=0.4)
    nt_subgraph.show(save_file)
