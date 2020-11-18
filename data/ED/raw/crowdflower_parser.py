import pandas as pd
import re
from contraction_map import CONTRACTION_MAP_a

from tqdm import tqdm

import string

import spacy
nlp = spacy.load('en_core_web_sm')

from wordsegment import load, segment
load()

from autocorrect import Speller
spell = Speller()

import preprocessor as p

from pycontractions import Contractions
cont = Contractions(api_key="glove-twitter-100")
cont.load_models()

df = pd.read_csv('text_emotion_crowdflower.csv')

print(df.head(5))

p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)

def parse(tweet):
    #print('original:')
    #print(tweet)
    tweet = p.clean(tweet)
    #print('tweet preprocessing:')
    #print(tweet)
    tweet =  list(cont.expand_texts([tweet], precise=True))[0]
    #print('contractions expanded:')
    #print(tweet)
    sent = []
    tweet = nlp(tweet)
    for word in tweet:
        if word.text in string.punctuation or word.text == '\'s' or word.text == '' or word.text == '\'':
            continue
        else:
            sent.append(spell(word.text))
    tweet = ' '.join(sent)
    #print('spell check:')
    #print(tweet)
    return tweet

def logged_apply(g, func, *args, **kwargs):
    """
        func - function to apply to the dataframe
        *args, **kwargs are the arguments to func
        The method applies the function to all the elements of the dataframe and shows progress
    """
    step_percentage = 100. / len(g)
    import sys
    print('\rApply progress:   0%', end = "")
    sys.stdout.flush()

    def logging_decorator(func):
        def wrapper(*args, **kwargs):
            progress = wrapper.count * step_percentage
            #sys.stdout.write('\033[D \033[D' * 4 + format(progress, '3.0f') + '%')
            print('\rApply progress:   {prog}%'.format(prog=progress), end = "")
            sys.stdout.flush()
            wrapper.count += 1
            return func(*args, **kwargs)
        wrapper.count = 0
        return wrapper

    logged_func = logging_decorator(func)
    res = g.apply(logged_func, *args, **kwargs)
    print('\rApply progress:   1 00%', end = "")
    sys.stdout.flush()
    print('\n')
    return res

df['parsed tweets'] = logged_apply(df['content'], parse)
df.to_csv('parsed_crowdflower_dataset.csv')
