import networkx as nx
import pandas as pd
import pickle
import os
from nltk.corpus import reuters
from tqdm import tqdm

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')
file = 'adjacency_matrix.pickle'
pkl = open(file, 'rb')
G = pickle.load(pkl)
pkl.close()

import string

V = list(G.nodes)
corpus = pd.DataFrame(index = V, columns = reuters.fileids())
corpus = corpus.fillna(0)
def create_corpus():
    for f in tqdm(reuters.fileids()):
        doc = reuters.words(f)
        doc = ' '.join(doc)
        doc = nlp(doc)
        for token in doc:
            if token.text in string.punctuation or token.text == '\'s' or token.text == '':
                continue
            try:
                token = token.lemma_
                corpus.loc[token][f]+=1
            except KeyError:
                pass
create_corpus()
file = 'corpus.pickle'
pkl = open(file, 'wb')
pickle.dump(corpus, pkl)
pkl.close()






