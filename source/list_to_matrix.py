import pandas as pd
import networkx as nx
import os
import pickle
import ast
from raw_data import Data
from graph import graph

file = 'adjacency_list.pickle' 
#df = pd.read_csv(file)
#print(df)
#df = df.iloc[:,1: ]
#df =df.dropna()
pkl = open(file, 'rb')
df = pickle.load(pkl)
pkl.close()
class convert:
    def __init__(self, path):
        self.path = path
        self.data = Data(path = self.path)  
        self.dict = Data.comple_dictionary()
        self.g = graph(self.dict).construct_graph()
        #pkl.close()
        #logged_apply(self.df.iloc[:, 0], self.edges_)


    def edges_(self, w):
        #print(df[df.iloc[:,0]==w].iloc[:,1].iloc[0])
        def_w_list = self.df[self.df.iloc[:, 0] == w].iloc[:, 1].iloc[0]
        edges = []
        for def_w in def_w_list:
            edges.append((w, def_w))
        self.g.add_nodes_from([w])
        #print(len(g.nodes))
        #print(edges)
        self.g.add_edges_from(edges)

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
print(len(df.iloc[:,0]))
logged_apply(df.iloc[:,0], edges_)
file = 'adjacency_matrix.pickle'
pkl = open(file, 'wb')
pickle.dump(g, pkl)
pkl.close()

