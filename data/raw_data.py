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
        self.data = self.data.iloc[1:3]
        print(self.data.head())
        self.data.columns = [0,1]
        self.data.fillna('')
        #print(self.data.head())
        #self.random_state = random_state
        #self.sample = self.data.sample(frac = 0.1, random_state = self.random_state)
        #self.sample = self.sample.set_index(np.arange(len(self.sample)))
                
    def remove_en_NaN(self):
        self.data = self.data[self.data.iloc[:,0] != 'en'][self.data.iloc[:, 1] != 'en']
        self.data.to_csv(self.path, header = None, encoding = 'latin1')

    def unique_word_keys(self):
        """
            This function returns all the unique word keys in wiktionary dumps
            output - (numpy.ndarray, int)
        """
        keys = self.sample[0].unique()
        num_keys = len(keys)
        return (keys, num_keys) 

    def unique_words_definitions(self):
        """
            This methods returns the unique words in the definitions
            output - (numpy.ndarray, int)
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
        self.data.iloc[:,1] = self.logged_apply(self.data.iloc[:,1], self.definition)
        return self.data

    def logged_apply(self, g, func, *args, **kwargs):
        """
            g - dataframe
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
#data.remove_en_NaN()
#print(data.data.shape)
#unique_words_def_file = '/home/shandilya/Shandilya/Padhai/CS6251/Project/data/unique_words_def_1.pickle'
#unique_words_keys_file = '/home/shandilya/Shandilya/Padhai/CS6251/Project/data/unique_words_keys_1.pickle'
dictionary = 'dictionary.pickle'
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
   

