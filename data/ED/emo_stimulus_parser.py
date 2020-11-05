import pandas as pd
from bs4 import BeautifulSoup as BS
from tqdm import tqdm

import spacy
nlp = spacy.load('en_core_web_sm')

import string

txt = open('Dataset/No Cause.txt', 'r').readlines()

emotions = ['happy', 'anger', 'fear', 'surprise', 'disgust', 'shame', 'sad']

data = []

for line in tqdm(txt):
    soup = BS(line, features="lxml")
    for emo in emotions:
        sent = soup.find(emo)
        if sent != None:
            sent = nlp(sent.text)
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
