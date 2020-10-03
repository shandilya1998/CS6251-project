import numpy 
from vertex import Vertex

class Graph:
    def __init__(self, adjacency_list, word2index, index2word):
        self.word2index = word2index
        self.index2word = index2word
        self.adjacency_list = adjacency_list

    def __getitem__(self, key):
        return self.adjacency_list[key]        
