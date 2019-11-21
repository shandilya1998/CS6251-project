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
   

path = 'wordlist_tokenized.csv'
df = pd.read_csv(path)
df = df.iloc[:, [0,2]]
# Import combined wordlist
path = 'combined_wordlist.pickle'
pkl = open(path, 'rb')
df2 = pickle.load(pkl)
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
        self.graph = pd.DataFrame(columns = [0,1])
        self.m_graph = pd.DataFrame()
        self.wordlist = []
        
    def combine_polysemous_definitions(self):
        unique_keys = pd.Series(self.data.iloc[:, 0].unique())
        definitions = self.logged_apply(unique_keys, self.concat_definition)
        self.data = pd.concat([unique_keys, definitions], axis = 1)
        #print(self.data)
        pkl = 'combined_wordlist.pickle'
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
            def_w =  self.data[self.data[0] == w][1].iloc[0]
            for df in def_w:
                df = ast.literal_eval(df)
                for w_ in df:
                    self.add_edge((w, w_))
        return self.graph

    def m_construct_graph(self):
        for w in tqdm(self.data.iloc[:, 0].values):
            self.m_add_vertex(w)
            def_w = self.data[self.data[0] == w][1].iloc[0]
            for df in def_w:
                df = ast.literal_eval(df)
                for w_ in df:
                    self.m_add_edge((w, w_))
        return self.m_graph

    def vertices(self):
        """ 
            returns the vertices of a graph 
        """
        return list(self.graph.iloc[:,0].values)

    def edges(self):
        """ 
            returns the edges of a graph 
        """
        return self.__generate_edges()

    def add_vertex(self, w):
        """ 
            If the word "word" is not in
            self.graph, a key "word" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if w not in self.graph.iloc[:,0]:    
            self.graph = self.graph.append(pd.DataFrame([[w, []]]), ignore_index = True)
            #print(self.graph)

    def m_add_vertex(self, w):
        """
            If the word w is not in self.wordlist, 
            a key w is added to self.wordlist. This self.wordlist
            contains the row and column names for the 
            adjacecny matrix self.m_graph
        """
        if w not in self.wordlist:
            self.wordlist.append(w)
            self.m_graph = self.m_graph.append(pd.DataFrame([[1]],index = [w], columns = [w]))
            
    def add_edge(self, edge):
        """ 
            assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        (w, def_w) = edge
        self.add_vertex(def_w)
        if w in self.graph.iloc[:, 0].values:
            if def_w not in self.graph[self.graph[0] == w][1].iloc[0]:
                self.graph[self.graph[0] == w][1].iloc[0].append(def_w)
        else:
            self.add_vertex(w)
            self.graph[self.graph[0] == w][1].iloc[0].append(def_w)

    def m_add_edge(self, edge):
        """
            edge : tuple
        """
        w, def_w = edge
        self.m_add_vertex(def_w)
        if w in self.wordlist:
            #print(self.m_graph)
            if self.m_graph[w][def_w] == np.nan:
                self.m_graph[w][def_w] = 1
            else:
                self.m_graph[w][def_w]+=1
        else:
            self.m_add_vertex(w)
            self.m_graph[w][def_w] = 1

    def __generate_edges(self):
        """ 
            A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one (a loop back to the vertex) or two
            vertices
        """
        edges = []
        for vertex in self.graph.iloc[:,0].values:
            for neighbour in self.graph[self.graph[0] == vertex][1].iloc[0]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

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

#ob =  graph(df)
#graph = graph(df2).construct_graph()
m_graph = graph(df2).m_construct_graph()
