import pandas as pd
import numpy as np 
import os
from tqdm import tqdm
import pickle
#from data.graph import graph_v1, sample
import networkx as nx

file = '../data/adjacency_matrix.pickle'
pkl = open(file, 'rb')
G = pickle.load(pkl)
pkl.close()

class SCC():
    """
        A directed graph is strongly connected if there is a path between all pairs of vertices.
    """
    def __init__(self, G):
        self.G = nx.from_pandas_adjacency(G)

    def get_SCC_generator(self, G = self.G):
        scc = nx.strongly_connected_component_subgraphs(G, copy=True)
        return scc


class stack():
    def __init__(self):
        self.stack = []

    def pop(self):
        return self.stack.pop()

    def push(self, item):
        return self.append(item)

scc = SCC(G).get_SCC_generator() # G is a pandas adjacecny matrix
file = '../data/scc.pickle'
pkl = open(file, 'wb')
pickle.dump(scc, pkl)
pkl.close()
