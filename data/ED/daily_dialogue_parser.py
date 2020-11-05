import pandas as pd
import os
from tqdm import tqdm

import string

import spacy
nlp = spacy.load('en_core_web_sm')


txt = open('EMNLP_dataset/dialogues_text.txt', 'r').readlines()
emotions = open('EMNLP_dataset/dialogues_emotion.txt', 'r').readlines()

data = []
for txt, emo in tqdm(zip(txt, emotions)):
    line = [nlp(sent) for sent in txt.split('__eou__')[:-1]]
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
df.to_csv('dialy_dialog_dataset_parsed.csv', index = False)      
