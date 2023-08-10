# -*- coding: utf-8 -*-
""" Run English grammar on the text

Before running: make sure local server to call the Propbank grammar is running
The API url should be http://127.0.0.1:1170/extract-frames, or similar
"""
import argparse
import multiprocessing as mp
import psutil
from tqdm import tqdm
from src.logger import Logger

import requests
from settings import API_PROPBANK_GRAMMAR
from src.helpers import read_csv

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

def call_grammar(text):
    """ call propbank grammar + error handling """
    try:
        return call_propbank_grammar(text)
    except Exception as exception:
        print(f"Exception caught for {text}")
        return {"statusCode": "500", "errorMessage": exception}


def main(values):
    # results = []
    # for i in tqdm(range(len(values))):
    #     results.append(call_grammar(values[i]))
    # values = [()]
    with mp.Pool(processes=psutil.cpu_count()) as pool:
        # results = []
        # for result in tqdm(pool.map(call_grammar, values), total=len(values)):
        #     results.append(result)

        results = list(tqdm(pool.imap(call_grammar, values), total=len(values)))

        pool.close()
        pool.join()
    return results

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-df', "--dataframe", required=True,
                    help="Path to pandas dataset containing the data")
    ap.add_argument('-ci', "--column_input", required=True,
                    help="Column to extract the values from")
    ap.add_argument('-co', "--column_output", required=True,
                    help="Column to save the values to in enriched df")
    ap.add_argument('-o', '--output', required=True,
                    help="Output pandas path to save enriched df")
    args_main = vars(ap.parse_args())

    LOGGER = Logger()
    LOGGER.log_start(name="Extracting Propbank Rolesets")
    DF_DATA = read_csv(args_main["dataframe"])
    VALUES = DF_DATA[args_main["column_input"]].values
    RESULTS = main(VALUES)

    DF_DATA[args_main["column_output"]] = RESULTS
    DF_DATA = DF_DATA[[col for col in DF_DATA.columns if col != 'Unnamed: 0']]
    DF_DATA.to_csv(args_main["output"])
    LOGGER.log_end()
