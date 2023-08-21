# Narratives from tweets

This is the code for the paper submitted to K-CAP 2023: "OKG: A Knowledge Graph for Social Media Discourse Analysis on Inequality".

First clone the repo
```bash
git clone https://github.com/SonyCSLParis/narratives_from_tweets
```

## 1. Set Up


### Virtualenv
Python version used: 3.10.6. We recommend to use a virtual environment.

Install the requirements:
```bash
pip install -r requirements.txt
```

### Babel

To install Babel (necessary to extract frames from the PropBank grammar), see and check guide from [here](https://gitlab.ai.vub.ac.be/ehai/babel).

To start the PropBank server, run in your Common Lisp editor:
```common lisp
(load "/path/to/babel-development/grammars/propbank-grammar/web-service/start-server.lisp")
```

### Endpoint

Possible to use the public API endpoint for Framester, but we strongly recommend to set up a local repository. You can download the data on [their website](https://framester.github.io). We used GraphDB to set up the local endpoint.


### src/private.py 
Create a `src/private.py` file, and add the following variables:
* `FOLDER_PATH`: path to the local repository
* `API_PROPBANK_GRAMMAR`: API to call the PropBank grammar (by default: `http://127.0.0.1:1170/extract-frames`)
* `SPARQL_ENDPOINT`: endpoint to Framester

### Install

Run the following for setting up the packages
```bash
python setup.py install
python -m spacy download en_core_web_sm
```

### Troubleshooting

- If you work on an Apple Silicon Machine + conda, you might later be prompted to download again `grpcio`, you can do it using:
    ```bash
    conda install grpcio=1.43.0 -c conda-forge
    ```

## 2. Ontology Documentation

We used the [Widoco](https://github.com/dgarijo/Widoco) Wizard for documenting our ontology.

The full HTML documentation can be found in `ontology/observatory/index-en.html`.

## 3. Pipeline

The `src/main.py` file runs all the components in the pipeline.

1. **Build KG directly from the triples**: `src/build_kg/build_kg_from_triples.py`
2. **Extract descriptions (tweet content)**: `src/features/get_description_pred.py`
3. **Pre-processing**: `src/features/pre_process.py`
4. **Extract frames with PropBank grammar**: `src/features/call_pb_grammar.py`
5. **Extract other features (sentiment, etc)**: `src/features/extract_other_features.py`
6. **Build graph from PropBank output + metrics**: `src/build_kg/build_kg_from_pb.py`
7. **Build the graph that links nif:Structure nodes**: `src/build_kg/add_super_string.py`

To run the pipeline, you need to have a `.csv` file with your triples, or a folder with such `.csv`.

If you have one `.csv` file:
```python
python src/main.py -p <input-event-csv-file> -o <output-folder>
```

If you have a folder of `.csv` file:
```python
python src/main.py -f <input-event-folder-csv-file> -o <output-folder>
```