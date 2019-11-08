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
                 path = '/home/shandilya/Shandilya/Padhai/CS6251/Project/data/wordlist.csv'):
        self.path = path
        self.data = pd.read_csv(self.path, encoding = 'latin1', header = None)
        #print(self.data.head())
        #print(self.data.columns)
        self.data = self.data.iloc[:, 1 :]
        self.data.columns = np.arange(self.data.shape[1])
        self.data.fillna('')
        #print(self.data.head())
        self.random_state = random_state
        self.sample = self.data.sample(frac = 0.1, random_state = self.random_state)
        self.sample = self.sample.set_index(np.arange(len(self.sample)))
                
    def remove_en_NaN(self):
        self.data = self.data[self.data.iloc[:,0] != 'en'][self.data.iloc[:, 1] != 'en']
        self.data.to_csv(self.path, header = None, encoding = 'latin1')

    def unique_word_keys(self):
        """
            This function returns all the unique word keys in wiktionary dumps
            output - (numpy.ndarray, int)
        """t
        keys = self.sample[0].unique()
        num_keys = len(keys)
        return (keys, num_keys) 

    def unique_words_definitions(self):
        """
            This methods returns the unique words in the definitions
            output - (numpy.ndarray, int)_
        """
        keys = np.ndarray((1, 2))
        definitions = self.sample[1].apply(str.split, ' ')
        for i in tqdm(range(0, len(definitions))):
            W = self.sample.iloc[i, 0]
            definition = pd.Series(definitions[i]).values
            for j in range(0, len(definition)):
                if definition[j] in keys[:,0]:
                    keys[j, 1]+=1
                else:
                    keys = np.append(keys,
                                          pd.DataFrame([definition[j],
                                                        0]).iloc[:,
                                                                 0].values.reshape((1,
                                                                                    2)), axis = 0)
        return (keys, len(keys))

    def definition(self, definition):
        try:
            np.isnan(definition)
        except TypeError:
            definition = nlp(definition)
            tokens = []
            for token in definition:
                if token.text in string.punctuation or token.text == '\'s' or token.text == '':
                    continue
                if token.is_stop:
                    continue
                token = token.lemma_
                tokens.append(token)
            definition = copy.deepcopy(tokens)
        return definition

    def compile_dictionary(self):
        for i in tqdm(range(1,self.data.shape[1])):
            self.data.iloc[:,i] = self.data.iloc[:,i].apply(self.definition)
        dictionary = pd.DataFrame(columns = ['word', 'definition'])
        for i in tqdm(range(len(self.data))):
            definitions = self.data.iloc[i, 1:]
            definitions = definitions.dropna().values 
            dictionary = dictionary.append(pd.DataFrame([[self.data.iloc[i, 0], definitions]], columns = ['word', 'definition']))
        return dictionary

data = Data()
#data.remove_en_NaN()
#print(data.data.shape)
unique_words_def_file = '/home/shandilya/Shandilya/Padhai/CS6251/Project/data/unique_words_def_1.pickle'
unique_words_keys_file = '/home/shandilya/Shandilya/Padhai/CS6251/Project/data/unique_words_keys_1.pickle'
dictionary = '/home/shandilya/Shandilya/Padhai/CS6251/Project/data/dictionary.pickle'
dict_ = data.compile_dictionary()
pkl = open(dictionary, 'wb')
pickle.dump(dict_, pkl)
pkl.close()
#pkl = open(unique_words_def_file, 'wb')
#pickle.dump(data.unique_words_definitions(), pkl)
#pkl.close()
#pkl = open(unique_words_keys, 'wb')
#pickle.dump(data.unique_word_keys(), pkl)
#pkl.close()
   

