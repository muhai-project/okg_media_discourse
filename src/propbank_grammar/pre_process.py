# -*- coding: utf-8 -*-
""" Preprocess text for extracting propbank role sets
- preprocessing pattern: from chatgpt

Steps in preprocessing
"""
import re
import argparse
from nltk.tokenize import sent_tokenize
import pandas as pd
from src.helpers import read_csv
from src.logger import Logger

class PreProcessor:
    """ Preprocessing text to run the English grammar 
    1. Cleaning text
    2. Sentence splitting """
    def __init__(self):
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
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.MULTILINE)

        self.replace_pattern = [
            ('#', ' '), ('\n', '. '), (': ', '. '), (' .', '.'), ('’', "'"),
            ('&amp;', 'and'), ('&lt;', 'less than'), ('&gt;', 'greater than '),
            ('&le;', 'less-or-equal than'), ('&ge;', 'greater-or-equal than'),
            ('"', ''), ("…", "."), ("“", ""),
            ('‘', "'"), ("*", ""), ("...", ".")]
        self.pipeline = [self.replace_regex, self.replace_simple]

    @staticmethod
    def replace_if_starts(text):
        """ Remove tag if starts by @{id} """
        pattern = r'^@[^\s]+'
        if re.match(pattern, text):
            return re.sub(pattern, '', text, count=1)
        return text

    def replace_simple(self, text):
        """ Simple replacements """
        for old, new in self.replace_pattern:
            text = text.replace(old, new)
        return text

    def replace_regex(self, text):
        """ Regex replacements """
        text = re.sub(self.emoji_pattern, '', text)  #emojis
        text = re.sub(self.url_pattern, '', text)  #urls
        text = re.sub(r"RT @[^:]+:", '', text)  #RT @{id}: [...]
        return text

    def clean_text(self, text):
        """ 1. Clean Text """
        text = text[1:] if text.startswith("\n") else text
        text = text[:-1] if text.endswith("\n") else text
        text = self.replace_if_starts(text)

        for func in self.pipeline:
            text = func(text.strip())
        return text
    
    @staticmethod
    def encode_utf8(text):
        """ Encoding to be readable by Propbank grammar"""
        return text.encode('ascii', 'ignore').decode()

    def __call__(self, input_df: pd.DataFrame, col_text: str = "object"):
        input_df["clean_text"] = input_df[col_text].apply(self.clean_text)

        output_df = {'subject': [], 'object': [], 'clean_text': [], 'sentence': [], 'sentence_id': []}
        for _, row in input_df.iterrows():
            sentences = sent_tokenize(row.clean_text)
            nb_sentences = len(sentences)

            for col in ["subject", "object", "clean_text"]:
                output_df[col] += [row[col]] * nb_sentences
            output_df['sentence'] += sentences
            output_df['sentence_id'] += [i for i in range(nb_sentences)]
        output_df = pd.DataFrame(output_df)
        output_df["sentence_utf8"] = output_df["sentence"].apply(self.encode_utf8)
        return output_df


if __name__ == '__main__':
    PRE_PROCESSOR = PreProcessor()
    LOGGER = Logger()
    # TEXT = """
    # RT @MVS__11: I will never understand why the military is always brought into the conversation when referring to inequality. Can someone exp… 📖📖

    # We've made our video of the talk between @Elif_Safak and @afuahirsch on How to Challenge Inequality available to all on our YouTube channel. ""

    # Check out this link: https://www.example.com. It's a great website!
    # """

    # NEW_TEXT = PRE_PROCESSOR(TEXT)
    # print(f"{TEXT}\n=====\n{NEW_TEXT}\n====={sent_tokenize(NEW_TEXT)}")

    ap = argparse.ArgumentParser()
    ap.add_argument('-p', "--path", required=True,
                    help=".csv file with descriptions, cf. get_description_pred")
    ap.add_argument('-o', "--output", required=False,
                    help="output .csv for saving")
    args_main = vars(ap.parse_args())

    LOGGER.log_start(name="Preprocessing tweets and chunking into sentences")
    DF_ = read_csv(args_main["path"])
    DF_ = PRE_PROCESSOR(input_df=DF_)
    if args_main["output"]:
        DF_.to_csv(args_main["output"])
    LOGGER.log_end()
