# -*- coding: utf-8 -*-
""" Running API on a list of values with multiprocessing """
import psutil
import argparse
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from src.call_api import call_propbank_grammar_api

def main(values):
    with mp.Pool(processes=psutil.cpu_count()) as pool:
        results = []
        for result in tqdm(pool.map(call_propbank_grammar_api, values),
                           total=len(values)):
            results.append(result)

        pool.close()
        pool.join()
    return [result.text for result in results]

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

    df_data = pd.read_csv(args_main["dataframe"])
    values = df_data[args_main["column_input"]].values
    results = main(values)

    df_data[args_main["column_output"]] = results
    df_data = df_data[[col for col in df_data.columns if col != 'Unnamed: 0']]
    df_data.to_csv(args_main["output"])
