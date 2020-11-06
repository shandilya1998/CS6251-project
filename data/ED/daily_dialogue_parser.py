import re
import pandas as pd
import os
from tqdm import tqdm
from contraction_map import CONTRACTION_MAP_a

import string

import spacy
nlp = spacy.load('en_core_web_sm')


txt = open('EMNLP_dataset/dialogues_text.txt', 'r').readlines()
emotions = open('EMNLP_dataset/dialogues_emotion.txt', 'r').readlines()

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


data = []
for txt, emo in tqdm(zip(txt, emotions)):
    line = [nlp(expand_contractions(sent)) for sent in txt.split('__eou__')[:-1]]
    emotion = [ int(e) for e in emo.split()]
    for l, em in zip(line, emotion):
        sent = ''
        for word in l:
            if word.text in string.punctuation or word.text == '\'s' or word.text == '' or word.text == '\'':
                continue
            else:
                sent = sent + word.text.lower() + ' '
        sent = sent.strip()
        data.append([sent, em])

df = pd.DataFrame(data, columns = ['sentences', 'emotion'])
df.to_csv('daily_dialog_dataset_parsed.csv', index = False)      
