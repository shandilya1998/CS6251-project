import pandas as pd
import re
from contraction_map import CONTRACTION_MAP_a

from tqdm import tqdm

import string

import spacy
nlp = spacy.load('en_core_web_sm')

from wordsegment import load, segment
load()

from hunspell import Hunspell
h = Hunspell()



def expand_contractions(text, contraction_mapping=CONTRACTION_MAP_a):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


df = pd.read_csv('text_emotion_crowdflower.csv')

print(df.head(5))

def parse(tweet):
    tweet = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.?&//=]*)', '', tweet, flags=re.MULTILINE)
    
    tweet = re.sub(r'[-a-zA-Z0–9@:%._\+=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.?&//=]*)', '', tweet, flags=re.MULTILINE) # to remove other url links
    
    tweet = ''.join(re.sub("(@[A-Za-z0–9]+)|([0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
    
    tweet = ' '.join(segment(tweet)) 
    tweet = nlp(expand_contractions(tweet))
    sent = []
    for word in tweet:
        if word.text in string.punctuation or word.text == '\'s' or word.text == '' or word.text == '\'':
            continue
        else:
            sent.append(word.text.lower())
    tweet = []
    for word in sent:
        if not h.spell(word):
            try:
                word = h.suggest(word)[0]
            except:
                pass
        tweet.append(word)
    tweet = ' '.join(tweet)
    print(tweet)
    return tweet

df['parsed tweets'] = df['content'].apply(parse)
df.to_csv('parsed_crowdflower_dataset.csv')
