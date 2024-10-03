# OKG Pipeline

This is the code for the paper accepted for publication to K-CAP 2023: "OKG: A Knowledge Graph for Social Media Discourse Analysis on Inequality".

--- 


# Acknowledgement 
This work was funded by the European MUHAI project (Horizon 2020 research and innovation program) under grant agreement number 951846,  the Sony Computer Science Laboratories-Paris, the Vrije Universiteit Amsterdam, the University of Bremen, and the Venice International University. C.S. acknowledges financial support from PON R\&I 2014–2020 (FSE REACT-EU). We thank Frank van Harmelen, Annette ten Teije and Ilaria Tiddi for fruitful discussions.
# Application domain 
Graphs, Natural language processing, Semantic web.
# Citation
```citation
@inproceedings{blin2023okg,
  title={OKG: A Knowledge Graph for Fine-grained Understanding of Social Media Discourse on Inequality},
  author={Blin, In{\`e}s and Stork, Lise and Spillner, Laura and Santagiustina, Carlo},
  booktitle={Proceedings of the 12th Knowledge Capture Conference 2023},
  pages={166--174},
  year={2023}
}
```
# Code repository 
https://github.com/muhai-project/okg_media_discourse
# Contact: 
Inès Blin
# Contributors: 
Inès Blin

Lise Stork

Laura Spillner

Carlo Santagiustina
# Creation date 
17-01-2023
# Description 
This is the code for the paper accepted for publication to K-CAP 2023: "OKG: A Knowledge Graph for Social Media Discourse Analysis on Inequality". It enables to build a KG from a set of tweets and its metadata.
# DOI 
https://doi.org/10.1145/3587259.3627557
# Full title
okg_media_discourse
# Installation instructions 
First clone the repo
```bash
git clone https://github.com/muhai-project/okg_media_discourse
```
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

# Invocation 
To run the pipeline, you need to have a `.csv` file with your triples, or a folder with such `.csv`.

If you have one `.csv` file:
```python
python src/main.py -p <input-event-csv-file> -o <output-folder>
```

If you have a folder of `.csv` file:
```python
python src/main.py -f <input-event-folder-csv-file> -o <output-folder>
```

# License 
Apache License 2.0
# Name 
okg_media_discourse
# Ontologies
We used the [Widoco](https://github.com/dgarijo/Widoco) Wizard for documenting our ontology.

The full HTML documentation can be found in `ontology/observatory/index-en.html`.

To generate content from the Widoco software (from the `ontology` folder): 
```bash
java -jar widoco-1.4.19-jar-with-dependencies_JDK-17.jar -ontFile observatory.owl -outFolder obio -confFile config.properties -uniteSections
```
# Owner 
Inès Blin
# Owner type
User
# Programming languages 
Python
# Related papers 
OKG: A Knowledge Graph for Social Media Discourse Analysis on Inequality
# Repository Status 
Inactive
# Requirements 
Cf. `requirements.txt` for Python requirements.
# Scripts
Snippets of code contained in the repository
The `src/main.py` file runs all the components in the pipeline.
1. **Build KG directly from the triples**: `src/build_kg/build_kg_from_triples.py`
2. **Extract descriptions (tweet content)**: `src/features/get_description_pred.py`
3. **Pre-processing**: `src/features/pre_process.py`
4. **Extract frames with PropBank grammar**: `src/features/call_pb_grammar.py`
5. **Extract other features (sentiment, etc)**: `src/features/extract_other_features.py`
6. **Build graph from PropBank output + metrics**: `src/build_kg/build_kg_from_pb.py`
7. **Build the graph that links nif:Structure nodes**: `src/build_kg/add_super_string.py`# Support channels: 

