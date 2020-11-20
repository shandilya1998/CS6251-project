import pickle
import numpy as np
import math
import copy
import tensorflow as tf
from constants import *

import pandas as pd

import spacy
nlp = spacy.load('en_core_web_sm')

class DataLoader(tf.keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        train,
        train_test_split,
        dictionary_encoding = True,
        num_neighbours = 3,
        data = ED_DATA,
        adj = ADJ_LIST,
        wordlist = WORDLIST,
        index2word = INDEX2WORD,
        max_sent_length = MAX_SENT_LENGTH
    ):
        self.batch_size = batch_size
        self.data = pd.read_csv(data)
        if train:
            self.data=self.data.iloc[:math.floor(train_test_split*len(self.data))]
        else:
            self.data=self.data.iloc[math.floor(train_test_split*len(self.data)):]
        self.nn = num_neighbours
        self.g = pickle.load(open(adj, 'rb'))
        self.wordlist = pickle.load(open(wordlist, 'rb'))
        self.index2word = pickle.load(open(index2word, 'rb'))
        self.num_words = len(self.wordlist.keys())
        self.de = dictionary_encoding
        self.max_sent_length = max_sent_length

    def get_dict_vec(self, index):
        vec = np.zeros(self.num_words)
        value = 1
        vec[index] = 1/value
        queue = []
        queue.extend(self.g[index])
        value = 2
        i = 0
        flag = len(queue)
        while(len(queue)>1):
            index = queue.pop(0)
            vec[index] = 1/value
            queue.extend(self.g[index])
            i += 1
            if flag == i:
                flag = len(queue)
                i = 0
                #print(value)
                value += 1
            if value == self.nn+2:
                break
        return vec/np.sum(vec)

    def shuffle(self):
        self.data = self.data.sample(frac = 1).reset_index(drop=True)

    def __len__(self):
        return math.ceil(len(self.data)/self.batch_size)

    def __getitem__(self, idx):
        indices = range(self.batch_size*idx, self.batch_size*(1+idx))
        x = [[], []]
        y = []
        #print(idx)
        #print(indices)
        for i in indices:
            #print(i)
            row = self.data.iloc[0]
            sent = [self.get_dict_vec(int(val)) for val in row['word index seq'].split(' ')]
            pos_seq = [int(val) for val in row['pos_seq'].split(' ')]
            if len(sent)>self.max_sent_length:
                sent = sent[:self.max_sent_length]
                pos_seq = pos_seq[:self.max_sent_length]
            elif len(sent)<self.max_sent_length:
                lst = [np.zeros(self.num_words) for i in range(self.max_sent_length-len(sent))]
                zeros = [0 for i in range(self.max_sent_length-len(sent))]
                sent.extend(lst)
                pos_seq.extend(zeros)
            x[0].append(np.array(sent, dtype = np.float32))
            x[1].append(np.array(pos_seq, dtype = np.int32))
            y.append([row['emotion']])
            #print(x)
        x[0] = np.array(x[0], dtype = np.float32)
        #print(x[0].shape)
        x[1] = np.array(x[1], dtype = np.int32)
        #print(x[1].shape)
        y = np.array(y)
        return x, y
