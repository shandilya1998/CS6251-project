import pandas as pd
from bs4 import BeautifulSoup as BS
from tqdm import tqdm
from contraction_map import CONTRACTION_MAP_b
import re

import spacy
nlp = spacy.load('en_core_web_sm')

import string

txt = open('Dataset/No Cause.txt', 'r').readlines()

emotions = ['happy', 'anger', 'fear', 'surprise', 'disgust', 'shame', 'sad']

data = []

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP_b):
    
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

expand_contractions("Y'all can't expand contractions I'd think")

for line in tqdm(txt):
    soup = BS(line, features="lxml")
    for emo in emotions:
        sent = soup.find(emo)
        if sent != None:
            sent = nlp(expand_contractions(sent.text))
            s = ''
            for word in sent:
                if word.text in string.punctuation or word.text == '\'s' or word.text == '' or word.text == '\'':
                    continue
                else:
                    s = s + word.text.lower() + ' '
            s = s.strip()
            data.append([s, emo])
            break
    
df = pd.DataFrame(data, columns = ['sentences', 'emotion'])
df.to_csv('emotion_stimulus_dataset_parsed.csv', index = False)
