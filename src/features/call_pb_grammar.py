# -*- coding: utf-8 -*-
""" Run English grammar on the text

Before running: make sure local server to call the Propbank grammar is running
The API url should be http://127.0.0.1:1170/extract-frames, or similar
"""
import os
import argparse
import multiprocessing as mp
import psutil
from tqdm import tqdm

import requests
from settings import API_PROPBANK_GRAMMAR
from src.helpers import read_csv, check_args, get_dask_df
from src.logger import Logger

def call_propbank_grammar(utterance: str) -> dict:
    """ API call """
    headers = {
        "Content-Type": "application/json"
    }
    data = """
    {
        "utterance": <utterance>,
        "package": "propbank-grammar",
        "grammar": "*restored-grammar*",
        "timeout": 120
    }""".replace("<utterance>", f'"{utterance}"')
    response = requests.post(API_PROPBANK_GRAMMAR,
                             headers=headers, data=data, timeout=120)
    return response.text

def call_grammar(text: str) -> dict:
    """ call propbank grammar + error handling """
    if not text:
        return {"statusCode": "500", "errorMessage": "text empty"}
    try:
        return call_propbank_grammar(text)
    except Exception as exception:
        print(f"Exception caught for {text}")
        return {"statusCode": "500", "errorMessage": exception}

def main(values: list[str]) -> list[str]:
    """ Main mp function"""
    with mp.Pool(processes=psutil.cpu_count()) as pool:

        results = list(tqdm(pool.imap(call_grammar, values), total=len(values)))

        pool.close()
        pool.join()
    return results

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', "--path", required=False,
                    help=".csv file with descriptions, cf. get_description_pred")
    ap.add_argument('-f', "--folder", required=False,
                    help="folder with .csv file with triple structure")
    ap.add_argument('-ci', "--column_input", required=True,
                    help="Column to extract the values from")
    ap.add_argument('-co', "--column_output", required=True,
                    help="Column to save the values to in enriched df")
    ap.add_argument('-o', '--output', required=True,
                    help="Output folder to save enriched df")
    args_main = vars(ap.parse_args())

    check_args(args=args_main)
    DFS = get_dask_df(args=args_main)

    LOGGER = Logger()
    LOGGER.log_start(name="Extracting Propbank Rolesets")

    RESULTS = []
    for DF_, I in DFS:
        I = I.replace(".csv", "").split("_")[1]
        save_path = os.path.join(args_main["output"], f"pb_{I}.csv")
        if not os.path.exists(save_path):  # Running from where it stopped (if script interrupted)
            DF_ = read_csv(DF_.compute())
            VALUES = DF_[args_main["column_input"]].values

            CURR_RES = main(VALUES)
            DF_[args_main["column_output"]] = CURR_RES
            DF_ = DF_[[col for col in DF_.columns if col != 'Unnamed: 0']]
            DF_.to_csv(save_path)
    LOGGER.log_end()
