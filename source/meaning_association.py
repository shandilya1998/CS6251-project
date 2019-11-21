import pandas as pd
import networkx as nx
import numpy as np 
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm

file = '../data/m_graph.pickle'
pkl = open(file, 'rb')
G = pickle.load(pkl)
pkl.close()

class meaning_association():
    def __init__(self, G):
        self.G
        self.G_ = nx.from_pandas_adjacency(G)
        self.m_def = pd.DataFrame()
        self.wordlist = []

    def get_meaning_matrix(self, M = self.m_meaning_association):
        M = self.logged_apply(M.iloc[:,1], self.create_matrix)
        return return M

    #def add_vertex(self, w):
    #    if w not in self.wordlist:
    #       self.m_def = self.m_def.append(pd.DataFrame([[1]], index = [w], columns = [w]))

    #def add_meaning_val(self, edge):
    #    w, def_w = edge
    #    self.add_vertex(def_w)
    #    if w in self.wordlist:
    #        if self.m_def[w][def_w] == np.nan:
    #            self.m_def[w][def_w] = self.meaning_association(w, def_w)
    #   else:
    #       self.add_vertex(w)
    #        self.m_def[w][def_w] = self.meaning_association(w, def_w)

    #def meaning_association(self, w, def_w):
    #    return 1.0

    def create_matrix(self):

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




    


