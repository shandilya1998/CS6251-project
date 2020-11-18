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

ADJ_LIST = '../../data/adjacency_list.pickle'
W2I = '../../data/wordlist.pickle'
I2W = '../../data/index2word.pickle'

class Data():
    def __init__(self,
                 random_state = 1,
                 path = '../../data/wordlist.csv'):
        self.path = path
        self.data = pd.read_csv(self.path, encoding = 'latin1',  header = None, engine = 'c' )
        self.data = self.data.iloc[:, 1:]
        self.data.fillna('')
        self.data.columns = [col-1 for col in self.data.columns]
        #print(self.data.columns)
        self.wordlist = set()
        self.mapping_wi = {}
        self.mapping_iw = {}
        self.adjacency_lst = {}
                
    def get_definition(self, definition):
        tokens = set()
        try:
            definition = nlp(definition)
            for token in definition:
                if token.text in string.punctuation or token.text == '\'s' or token.text == '' or token.text == ' ':
                    continue
                if token.is_stop:
                    continue
                token = token.lemma_
                tokens.add(token)
            return list(tokens)
        except TypeError:
            return []
    
    def compile_dictionary(self):
        """
            Applies the function definition() on the data
            returns transformed data
        """
        #print(self.data.head().iloc[:,1])

        self.data.iloc[:,1] = self.logged_apply(
            self.data.iloc[:,1],
                self.get_definition
            )

    def get_processed_data(self, compile_dict = True, compile_map = True):
        
        if compile_dict:
            print('Compiling Dictionary')
            self.compile_dictionary()
        
            if os.path.exists('../../data/dictonary.pickle'):
                os.remove('../../data/dictionary.pickle')
            self.data.to_pickle('../../data/dictionary.pickle')
        
        else:
            pd.read_pickle('../../data/dictionary.pickle')
    
        if compile_map:
            print('Creating Wordlist')
            for index, row in tqdm(self.data.iterrows()):
                self.wordlist.add(row[0])
                try:
                    for word in row[1]:
                        self.wordlist.add(word)
                except TypeError:
                    pass
            self.wordlist = list(self.wordlist)
            print('Creating Word to Index and Index to Word Maps')
            for i, word in tqdm(enumerate(self.wordlist)):
                self.mapping_wi[word] = i
                self.mapping_iw[i] = word

            if os.path.exists(W2I):
                os.remove(W2I)
            pkl = open(W2I, 'wb')
            pickle.dump(self.mapping_wi, pkl)
            pkl.close()

            if os.path.exists(I2W):
                os.remove(I2W)
            pkl = open(I2W, 'wb')
            pickle.dump(self.mapping_iw, pkl)
            pkl.close()

        else:
            pkl = open(W2I, 'rb')
            self.mapping_wi = pickle.load(pkl)
            pkl.close()
            pkl = open(I2W, 'rb')
            self.mapping_iw = pickle.load(pkl)
            pkl.close()
        
        print('Creating Empty Adjacency List')
        
        for index in tqdm(self.mapping_iw.keys()):
            self.adjacency_lst[index] = set()

        print('Populating Adjacency List')
        for index, row in tqdm(self.data.iterrows()):
            lst = set()
            try:
                for word in row[1]:
                    lst.add(self.mapping_wi[word])
            except TypeError:
                print(row[1])
                pass
            try:
                self.adjacency_lst[self.mapping_wi[row[0]]] = self.adjacency_lst[self.mapping_wi[row[0]]].union(lst)
            except KeyError:
                self.adjacency_lst[self.mapping_wi[row[0]]] = lst
        
        for word in self.adjacency_lst.keys():
            self.adjacency_lst[word] = list(self.adjacency_lst[word])

        if os.path.exists(ADJ_LIST):
            os.remove(ADJ_LIST)
        pkl = open(ADJ_LIST, 'wb')
        pickle.dump(self.adjacency_lst, pkl)
        pkl.close()
        print('done')
            
    def logged_apply(self, g, func, *args, **kwargs):
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

data = Data()
data.get_processed_data(True, True)

