import pandas as pd
from nltk.corpus import reuters

file = 'wordlist.csv'
df = pd.read_csv(file, encoding = 'latin1', header = None)
df = df.iloc[:,1:3]
df.columns = [0,1]
df.fillna('')

wordlist = pd.DataFrame()
w_ = df.iloc[:,0].values
cat = reuters.categories()

filelist = reuters.fileids()
name = 'wordlist'

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')

import pickle

from tqdm import tqdm

for c in tqdm(cat):
    files = reuters.fileids(c)
    wordlist = pd.DataFrame()
    for f in tqdm(files[:20]):
        words = reuters.words(f)
        words = nlp(' '.join(words))
        for word in tqdm(words):
            try:
                word = str.lower(word.lemma_)
            except:
                pass
            if word in w_:
                def_w = df[df[0]==word]
                wordlist = wordlist.append(def_w, ignore_index = True)
    fname = name + c + '.pickle'
    pkl = open(fname, 'wb')
    pickle.dump(wordlist, pkl)
    pkl.close()






    


