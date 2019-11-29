import pandas as pd
#from raw_data import data # data is the object of class Data, which loads wordlist.csv into a DataFrame
import os
import matplotlib.pyplot as plt
import copy 
import pickle
#from raw_data import unique_word_keys_file, unique_words_def_file
import ast
from tqdm import tqdm
import numpy as np
import networkx as nx
   

# Import combined wordlist
path = 'dictionary.pickle'
pkl = open(path, 'rb')
df2 = pickle.load(pkl)
pkl.close()
f = 'all_words.pickle'
pkl = open(f, 'rb')
wordlist = pickle.load(pkl)
pkl.close()
class graph():
    def __init__(self, data):
        """
            This class contains all methods required to create a graph from a csv with w 
            and corresponding def(w)
            The class object requires the path to the csv file as an input during initialization
        """
        self.data = data
        #self.combine_polysemous_definitions()
        #self.word = input('input word to evolve dictionary network for: ')
        #self.network = self.evolve_graph(self.word)
        self.graph = nx.DiGraph()
        #self.df = pd.DataFrame(columns = [0,1])
        #self.lst = []
        #logged_apply(self.data.iloc[:,0], self.func)
        #pkl = open(path, 'wb')
        #pickle.dump(df, pkl)
        #pkl.close()
        #self.wordlist = []
        #self.data = self.df
        
    def combine_polysemous_definitions(self):
        """
            This method creates a combined wordlist of all words in the dictionary
            Redundant method
            Deprecated
        """
        unique_keys = pd.Series(self.data.iloc[:, 0].unique())
        definitions = logged_apply(unique_keys, self.concat_definition)
        self.data = pd.concat([unique_keys, definitions], axis = 1)
        #print(self.data)
        pkl = '../data/combined_wordlist.pickle'
        pkl = open(pkl, 'wb')
        pickle.dump(self.data, pkl)
        pkl.close()

    def concat_definition(self, key):
        data = self.data[self.data.iloc[:,0] == key]
        definitions = data.iloc[:,1]
        definition = []
        for df in definitions:
            #print(df)
            definition.append(df)
            #print(type(df))
            #print(definition)
        #print(definition)
        return definition

    def construct_graph(self):
        for w in tqdm(self.data.iloc[:, 0].values):
            self.add_vertex(w)
            def_w =  list(self.data[self.data[0] == w][1].values)
            for df in def_w:
                for w_ in df:
                    self.add_edge((w, w_))
        return self.graph

    def vertices(self):
        """ 
            returns the vertices of a graph 
        """
        return self.graph.nodes

    def add_vertex(self, w):
        """ 
            If the word "word" is not in
            self.graph, a key "word" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if w not in self.graph.nodes:    
            self.graph.add_nodes_from([w])
            #print(self.graph)
            
    def add_edge(self, edge):
        """ 
            assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        (w, def_w) = edge
        self.add_vertex(def_w)
        if w in self.graph.nodes:
            if def_w not in self.graph.neighbors(w):
                self.graph.add_edges_from([edge])
        else:
            self.add_vertex(w)
            self.graph.add_edges_from([edge])

    def func(self, word):
        d = list(df2[df2[0]==word][1].values)
        if word not in self.lst:
            self.df = self.df.append(pd.DataFrame([[word, d]], columns = [0,1]), sort = False)
            self.lst.append(word)

def logged_apply(g, func, *args, **kwargs):
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

#ob =  graph(df)
#graph = graph(df2).construct_graph()
#pkl = open(path, 'wb')
#pickle.dump(df, pkl)
#pkl.close()
graph = graph(df2).construct_graph()
file = 'graph.pickle'
pkl = open(file, 'wb')
pickle.dump(graph, pkl)
pkl.close()
