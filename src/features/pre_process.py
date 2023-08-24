# -*- coding: utf-8 -*-
""" Preprocess text for extracting propbank role sets
- preprocessing pattern: from chatgpt

Steps in preprocessing
"""
import os
import argparse
import multiprocessing as mp
import re
import spacy
import psutil
from tqdm import tqdm
import pandas as pd
import dask.dataframe as dd
from src.helpers import read_csv
from src.logger import Logger
from src.helpers import check_args, get_dask_df

NLP = spacy.load("en_core_web_sm")

class PreProcessor:
    """ Preprocessing text to run the English grammar 
    1. Cleaning text
    2. Sentence splitting """
    def __init__(self, nlp = NLP):
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE)
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            re.MULTILINE)

        self.replace_pattern = [
            ('#', ' '), ('\n', '. '), (': ', '. '), (' .', '.'), ('’', "'"),
            ('&amp;', 'and'), ('&lt;', 'less than'), ('&gt;', 'greater than '),
            ('&le;', 'less-or-equal than'), ('&ge;', 'greater-or-equal than'),
            ('"', ''), ("…", "."), ("“", ""),
            ('‘', "'"), ("*", ""), ("...", ".")]
        self.pipeline = [self.replace_regex, self.replace_simple]
        self.nlp = nlp

    @staticmethod
    def replace_if_starts(text: str) -> str:
        """ Remove tag if starts by @{id} """
        pattern = r'^@[^\s]+'
        if re.match(pattern, text):
            return re.sub(pattern, '', text, count=1)
        return text

    def replace_simple(self, text: str) -> str:
        """ Simple replacements """
        for old, new in self.replace_pattern:
            text = text.replace(old, new)
        return text

    def replace_regex(self, text: str) -> str:
        """ Regex replacements """
        text = re.sub(self.emoji_pattern, '', text)  #emojis
        text = re.sub(self.url_pattern, '', text)  #urls
        text = re.sub(r"RT @[^:]+:", '', text)  #RT @{id}: [...]
        return text

    def clean_text(self, text: str) -> str:
        """ 1. Clean Text """
        text = text[1:] if text.startswith("\n") else text
        text = text[:-1] if text.endswith("\n") else text
        text = self.replace_if_starts(text)

        for func in self.pipeline:
            text = func(text.strip())
        return text

    @staticmethod
    def encode_utf8(text: str) -> str:
        """ Encoding to be readable by Propbank grammar"""
        return text.encode('ascii', 'ignore').decode()

    def get_token_mapping(self, text_1: str, text_2: str) -> dict:
        """ Should be: text_1: orig (longer), text_2: pre_processed (shorter)
        if not: returning inverse """
        if len(text_1) < len(text_2):
            return self.get_token_mapping(text_1=text_2, text_2=text_1)

        doc_1, doc_2 = self.nlp(text_1.strip()), self.nlp(text_2.strip())
        token_mapping = {}
        i_1, i_2 = 0, 0

        while i_2 < len(doc_2) and i_1 < len(doc_1):  # mapping all tokens from `text_2`
            if (doc_2[i_2].text == doc_1[i_1].text) or \
                (f"#{doc_2[i_2].text}" == doc_1[i_1].text):  # mapping found
                token_mapping[i_2] = i_1
                i_2 += 1
                i_1 += 1
            else:  # mapping not found, increasing i_1 by 1
                i_1 += 1
        return token_mapping

    def __call__(self, input_df: pd.DataFrame, col_text: str = "object") -> pd.DataFrame:
        output_df = {'subject': [], 'object': [], 'sent': [], 'sent_id': [], 'sent_len': []}

        for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
            doc = self.nlp(row.object)
            sentences = list(doc.sents)
            nb_sentences = len(sentences)

            for i, sent in enumerate(sentences):
                output_df['sent'].append(sent.text.replace("\n", ""))
                output_df['sent_id'].append(i)
                output_df['sent_len'].append(len(sent))

            for col in ["subject", "object"]:
                output_df[col] += [row[col]] * nb_sentences

        output_df = pd.DataFrame(output_df)
        tqdm.pandas()
        output_df['sent_clean'] = output_df["sent"].progress_apply(self.clean_text)
        output_df["sent_clean_utf8"] = output_df["sent_clean"].progress_apply(self.encode_utf8)
        output_df["sent_utf8"] = output_df["sent"].progress_apply(self.encode_utf8)
        output_df["sent_mapping"] = output_df.progress_apply(
            lambda row: self.get_token_mapping(row["sent_utf8"], row["sent_clean_utf8"]), axis=1)

        return output_df

def mp_func(des_df: dd.core.DataFrame) -> (pd.DataFrame, str):
    """ MP function: preprocessing """
    pre_processor = PreProcessor()
    df_ = read_csv(des_df[0].compute())
    return pre_processor(df_), des_df[1]

def main(dfs: list[dd.core.DataFrame]) -> list:
    """ Main multiprocessing function """
    with mp.Pool(processes=psutil.cpu_count()) as pool:
        results = []
        for result in tqdm(pool.map(mp_func, dfs),
                           total=len(dfs)):
            results.append(result)

        pool.close()
        pool.join()
    return results

if __name__ == '__main__':
    LOGGER = Logger()

    ap = argparse.ArgumentParser()
    ap.add_argument('-p', "--path", required=False,
                    help=".csv file with descriptions, cf. get_description_pred")
    ap.add_argument('-f', "--folder", required=False,
                    help="folder with .csv file with triple structure")
    ap.add_argument('-o', "--output", required=False,
                    help="output folder for saving")
    args_main = vars(ap.parse_args())

    check_args(args=args_main)
    DFS = get_dask_df(args=args_main)

    LOGGER.log_start(name="Preprocessing tweets and chunking into sentences")
    RES = main(dfs=DFS)

    for elt in RES:
        print(elt)

    # if args_main["output"]:
    #     for df_o, index in RES:
    #         index = index.replace(".csv", "").split("_")[1]
    #         df_o.to_csv(os.path.join(args_main["output"], f"pp_{index}.csv"), encoding="utf8")
    # LOGGER.log_end()
