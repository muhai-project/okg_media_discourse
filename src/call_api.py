# -*- coding: utf-8 -*-
""" Calling APIs and storing results """
import requests

def call_propbank_grammar_api(utterance: str) -> dict:
    headers = {
        "Content-Type": "application/json"
    }
    data = """
    {
        "utterance": <utterance>,
        "package": "propbank-grammar",
        "grammar": "*restored-grammar*",
        "timeout": 100
    }""".replace("<utterance>", f'"{utterance}"')
    response = requests.post('http://127.0.0.1:1170/extract-frames',
                             headers=headers, data=data, timeout=3600)
    return response

# utterance = "He told him a story"
# response = call_propbank_grammar_api(utterance=utterance)
# print(response.text)