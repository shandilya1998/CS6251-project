import pandas as pd
import numpy as np 
import os
from tqdm import tqdm
import pickle
from data.graph import graph_v1, sample

class SCC():
    """
        A directed graph is strongly connected if there is a path between all pairs of vertices.
    """
    def __init__(self, unique_words_path):
        self.adjacency_list = graph_v1(sample, unique_words_path).construct_adjacency_list()
        self.stack = stack()

    def getTranspose(self):
        graph = 

class stack():
    def __init__(self):
        self.stack = []

    def pop(self):
        return self.stack.pop()

    def push(self, item):
        return self.append(item)
