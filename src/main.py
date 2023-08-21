"""
Main script to convert the .csv triples into a KG

Input = either (a) a folder containing .csv files of triples (b) a simple .csv file

Different steps
1. Build graph from triples

2. Extract additional features
2.1. Extract description (text of tweets)
2.2. Pre-process
2.3. Extract frames from PropBank grammar
2.4. Extract other features (sentiment, subjectivity, etc)
2.5. Build graph

3. Build an additional KG that harmonizes the KGs from 1. and 2.

#1 Build graph from triples
python src/build_kg/build_kg_from_triples.py -f <folder-input> -o <folder-output>

#2 Build graph from propbank + other features

#2.1 Extract descriptions
python src/features/get_description_pred.py -f <folder-input> -o <folder-output> -pr 1

#2.2 Pre process
python src/features/pre_process.py -f <folder-input> -o <folder-output>

#2.3 Extract pb grammar
python src/features/call_pb_grammar.py -f <folder-input> \
    -ci sent_clean_utf8 -co propbank_output -o <folder-output>

#2.4 Extract other features
python src/features/extract_other_features.py -f <folder-input> -o <folder-output>

#2.5 Build graph
python src/build_kg/build_kg_from_pb.py -f <folder-input> -o <folder-output>

#3. Build superstring graph
"""

import os
import subprocess
import argparse
from datetime import datetime

def make_dirs(folder):
    """ Creating dirs """
    if not os.path.exists(folder):
        os.makedirs(folder)

    for sub_fold in ["descriptions", "feat", "from_triples", "kg_from_pb",
                     "pb", "pp", "superstring"]:
        curr_folder = os.path.join(folder, sub_fold)
        if not os.path.exists(curr_folder):
            os.makedirs(curr_folder)

def main(args):
    """ Running all steps of the pipeline """
    if args_main["path"]:
        arg_k = "-p"
        arg_val = args_main["path"]
    else:  # folder of .csv
        arg_k = "-f"
        arg_val = args_main["folder"]

    start = datetime.now()

    #1 Build graph from triples
    command = f"""
    python src/build_kg/build_kg_from_triples.py {arg_k} {arg_val} \
        -o {os.path.join(args["output"], "from_triples")}
    """
    subprocess.call(command, shell=True)

    #2 Build graph from propbank + other features

    #2.1 Extract descriptions
    command = f"""
    python src/features/get_description_pred.py {arg_k} {arg_val} \
        -o {os.path.join(args["output"], "descriptions")}
    """
    subprocess.call(command, shell=True)

    #2.2 Pre process
    command = f"""
    python src/features/pre_process.py \
        -f {os.path.join(args["output"], "descriptions")} \
            -o {os.path.join(args["output"], "pp")}
    """
    subprocess.call(command, shell=True)

    #2.3 Extract pb grammar
    command = f"""
    python src/features/call_pb_grammar.py \
        -f {os.path.join(args["output"], "pp")} \
            -o {os.path.join(args["output"], "pb")} \
                -ci sent_clean_utf8 -co propbank_output
    """
    subprocess.call(command, shell=True)

    #2.4 Extract other features
    command = f"""
    python src/features/extract_other_features.py \
        -f {os.path.join(args["output"], "pb")} \
            -o {os.path.join(args["output"], "feat")}
    """
    subprocess.call(command, shell=True)

    #2.5 Build graph
    command = f"""
    python src/build_kg/build_kg_from_pb.py \
        -f {os.path.join(args["output"], "feat")} \
            -o {os.path.join(args["output"], "kg_from_pb")}
    """
    subprocess.call(command, shell=True)

    #3. Build superstring graph
    command = f"""
    python src/build_kg/add_super_string.py \
        -o {os.path.join(args["output"], "superstring")} \
            -f1 {os.path.join(args["output"], "from_triples", "superstring_cands.txt")} \
                -f2 {os.path.join(args["output"], "kg_from_pb", "superstring_cands.txt")}
    """
    subprocess.call(command, shell=True)

    end = datetime.now()
    print(f"Took {end-start}")

if __name__ == '__main__':
    # Run all steps of the pipeline
    # To be run from root directory of the repo
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', "--path", required=False,
                    help=".csv file with triple structure")
    ap.add_argument('-f', "--folder", required=False,
                    help="folder with .csv file with triple structure")
    ap.add_argument('-o', "--output", required=True,
                    help="folder_output")
    args_main = vars(ap.parse_args())

    make_dirs(folder=args_main["output"])
    main(args=args_main)
