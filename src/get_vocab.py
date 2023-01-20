# -*- coding: utf-8 -*-
"""
Getting vocab from recognised entities in tweets
"""
import json
import argparse
from src.helpers import ent_to_uri, get_spacy_docs_from_bytes

def main(pkl_file, output_file):
    """ Getting entities from spacy pipeline data and saving it into json"""
    docs = get_spacy_docs_from_bytes(pkl_file=pkl_file)
    entities = [ent for doc in docs for ent in doc.ents if ent._.dbpedia_raw_result]
    vocab = dict(enumerate(
        list(set([ent_to_uri(ent) for ent in entities]))
    ))
    with open(output_file, "w", encoding='utf-8') as openfile:
        json.dump(vocab, openfile, indent=4)

if __name__ == '__main__':
    # python src/get_vocab.py -p ./sample-data/docs_spacy_ukraine_russia.pkl -o ./sample-data/vocab_ukraine_russia.json
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', "--pkl", required=True,
                    help=".pkl file containing the spacy pipeline data in bytes format")
    ap.add_argument('-o', '--output', required=True,
                    help="output .json file to save the output vocab")
    args_main = vars(ap.parse_args())

    main(pkl_file=args_main['pkl'], output_file=args_main['output'])

