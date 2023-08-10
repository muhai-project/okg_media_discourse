part on poetry

start propbank server
(load "/Users/ines/Projects/babel-development/grammars/propbank-grammar/web-service/start-server.lisp")

poetry not working with pykeen, run in virtual env
torch-ppr==0.0.8
pykeen==1.9.0
pyrdf2vec
nest_asyncio
gensim
transformers==4.31.0

problem dependency with streamlit
pip install dependency

~/Library/"Application Support"/pypoetry/venv/bin/poetry

m1+conda, need to download grpcio manually
```
pip uninstall grpcio (if applicable)

```

could not install tensorflow with poetry (problem with grpcio)
took the latest versions within range of possible versions for tensorflow (needed for ampligraph)
conda install -c conda-forge tensorflow==2.9.1

hdt==2.3
urllib==1.26.11
nltk==3.8.1
torch==1.13.1
textblob==0.17.1
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

```python
import nltk
nltk.download("punkt")
```

fasttext
sentence-transformers
pykeen


Preprocessing data
download spacy model: python -m spacy download en_core_web_sm