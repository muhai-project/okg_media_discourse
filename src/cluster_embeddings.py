# -*- coding: utf-8 -*-
""" Clustering pre-trained embeddings

Resources for implementation
- Binary embeddings: https://blog.ekbana.com/loading-glove-pre-trained-word-embedding-model-from-text-file-faster-5d3e8f2b8455
- Progress bar when reading file lines:
https://blog.nelsonliu.me/2016/07/30/progress-bars-for-python-file-reading-with-tqdm/

"""
import os
import json
import mmap
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_entities(df_path: str, col: str) -> list[str]:
    """ Extracting entities from pd dataframe
    df_path: path to pd dataframe
    col: column to extract values from """
    return [x.strip() for x in pd.read_csv(df_path)[col].values]

def get_num_lines(file_path: str) -> int:
    """ Number of lines from file"""
    openfile = open(file_path, "r+", encoding="utf-8")
    buf = mmap.mmap(openfile.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def load_embeddings(input_file: str, entities: list[str], output_folder: str):
    """ Select and store entities embeddings + vocabulary
    input_file: .txt file path with name of URI + embedding
    entities: list of entities we are interested in
    output_folder: saving folder """
    ent_nb = 0
    vocab, embeddings = {}, []
    filename, _ = os.path.splitext(os.path.basename(input_file))

    with open(input_file, "r", encoding="utf-8") as openfile:

        for line in tqdm(openfile, total=get_num_lines(input_file)):
            splitted = line.split()
            if splitted[0].strip() in entities:
                vocab[ent_nb] = splitted[0].strip()
                ent_nb += 1
                embeddings.append([float(val) for val in splitted[1:]])

    with open(os.path.join(output_folder, "vocab.json"), "w", encoding='utf-8') as openfile:
        json.dump(vocab, openfile, indent=4)

    np.save(os.path.join(output_folder, f"embedding-{filename}.npy"), np.array(embeddings))

    print(f"# of entities: {len(entities)}")
    print(f"# of entities with embeddings: {len(vocab)}")

def get_preferences_affinity_propagation(df_path: str, vocab: dict,
                                         ent_col: str = 'uri', nb_col: str = 'count') -> np.array:
    """
    Getting preference parameter for AffinityPropagation clustering
    df_path: path to pd dataframe with entities and count
    vocab: mapping array index to entity name
    ent_col: entity column in dataframe
    nb_col: number column in dataframe """
    df_info = pd.read_csv(df_path)
    scores = dict(zip(df_info[ent_col].values, df_info[nb_col].values))
    
    res = [scores[vocab[str(i)]] for i in range(len(vocab))]
    res = [x if x > 50 else -5 for x in range(len(vocab))]
    return np.array(res)


ENTITIES = get_entities(df_path="./sample-data/dbpedia_spotlight_entities.csv", col="uri")
INPUT_FILE = "./sample-data/rdf2vec/uniform.txt"
OUTPUT_FOLDER = "./sample-data"

load_embeddings(INPUT_FILE, ENTITIES, OUTPUT_FOLDER)

VOCAB = json.load(open("./sample-data/vocab.json", 'r', encoding='utf-8'))

# preference = get_preferences_affinity_propagation(
#     df_path = "./sample-data/dbpedia_spotlight_entities.csv",
#     vocab=vocab)


# from sklearn.cluster import AffinityPropagation
# clustering_algorithm = AffinityPropagation()
# test=np.load("./sample-data/embedding-pageRank.npy")
# clusters_output=clustering_algorithm.fit(test)
# db1_labels=clusters_output.labels_
# labels, counts = np.unique(db1_labels[db1_labels>=0], return_counts=True)
# clusters={x: [] for x in range(len(counts))}
# for i, cluster in enumerate(db1_labels):
#     clusters[cluster].append(VOCAB[str(i)])
# with open("test.json", "w", encoding="utf-8") as openfile:
#     json.dump(clusters, openfile, indent=4)

    