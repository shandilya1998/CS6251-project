import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pickle
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')
import copy

class Data():
    def __init__(self, 
                 random_state = 1, 
                 path = 'wordlist.csv'):
        self.path = path
        self.data = pd.read_csv(self.path, encoding = 'latin1', header = None)
        self.data = self.data.iloc[:,1:]
        print(self.data.head())
        self.data.columns = [col-1 for col in self.data.columns]
        self.wordlist = set()
        self.mapping_wi = {}
        self.mapping_iw = {}
        self.adjacency_lst = {}
                
    def get_definition(self, definition):
        """
            Converts a string input definition into a list of lemmatized defining words
        """
        if not np.isnan(definition):
            definition = nlp(definition)
            tokens = set()
            for token in definition:
                if token.text in string.punctuation or token.text == '\'s' or token.text == '':
                    continue
                if token.is_stop:
                    continue
                token = token.lemma_
                tokens.add(token)
                self.wordlist.add(token)
            return list(tokens)
        return np.nan

    def compile_dictionary(self):
        """
            Applies the function definition() on the data 
            returns transformed data
        """
        for i in self.data.columns:
            if(i != 0):
                self.data.iloc[:,i] = self.logged_apply(
                    self.data.iloc[:,i], 
                    self.get_definition
                )
        self.wordlist = list(self.wordlist)

    def get_processed_data(self):
        self.compile_dictionary()
        
        if os.path.exists('dictonary.pickle'):
            os.remove('dictionary.pickle')
        pkl = open('dictionary.pickle', 'wb')
        pickle.dump(self.data)
        pkl.close()
        
        for i in range(len(self.wordlist)):
            self.mapping_wi[self.wordlist[i]] = i
            self.mapping_iw[i] = self.wordlist[i]

        if os.path.exists('wordlist.pickle'):
            os.remove('wordlist.pickle')
        pkl = open('wordlist.pickle', 'wb')
        pickle.dump(self.mapping_wi)
        pkl.close()

        if os.path.exists('index2word.pickle'):
            os.remove('index2word.pickle')
        pkl = open('index2word.pickle', 'wb')
        pickle.dump(pkl)
        pkl.close()

        for index, row in self.data.iterrows():
            lst = set()
            try:
                lst = set(self.adjacency_lst[self.mapping_wi[self.data[index, 0]]])
            except KeyError:
                pass
            for i in self.data.columns:
                if i!=0:
                    for j in range(self.data[index, i]):
                        lst.add(self.mapping_wi[self.data[index, i][j]])
            self.adjacency_lst[self.mapping_wi[self.data[index, 0]]] = list(lst)
        if os.path.exists('adjacency_list.pickle'):
            os.remove('adjacency_list.pickle')
        pkl = open('adjacency_list.pickle', 'wb')
        pickle.dump(self.adjacency_lst)
        pkl.close()
            
    def logged_apply(self, g, func, *args, **kwargs):
        """
            :x
            func - function to apply to the dataframe
            *args, **kwargs are the arguments to func
            The method applies the function to all the elements of the dataframe and shows progress
        """
        step_percentage = 100. / len(g)
        import sys
        sys.stdout.write('apply progress:   0%')
        sys.stdout.flush()

        def logging_decorator(func):
            def wrapper(*args, **kwargs):
                progress = wrapper.count * step_percentage
                sys.stdout.write('\033[D \033[D' * 4 + format(progress, '3.0f') + '%')
                sys.stdout.flush()
                wrapper.count += 1
                return func(*args, **kwargs)
            wrapper.count = 0
            return wrapper

        logged_func = logging_decorator(func)
        res = g.apply(logged_func, *args, **kwargs)
        sys.stdout.write('\033[D \033[D' * 4 + format(100., '3.0f') + '%' + '\n')
        sys.stdout.flush()
        return res

data = Data()
data.get_processed_data()
