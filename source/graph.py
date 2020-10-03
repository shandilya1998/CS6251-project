import numpy 
from vertex import Vertex

ADJ_LIST = '../data/adjacency_list.pickle'
W2I = '../data/wordlist.pickle'
I2W = '../data/index2word.pickle'


class Graph:
    def __init__(self, 
        adjacency_list = ADJ_LIST, 
        word2index = W2I, 
        index2word = I2W 
    ):
        pkl = open(adjacency_list, 'rb')
        self.adjacency_list = pickle.load(pkl)
        pkl.close()

        pkl = open(word2index, 'rb')
        self.word2index = pickle.load(pkl)
        pkl.close()

        pkl = open(index2word, 'rb')
        self.index2word = pickle.load(pkl)
        pkl.close()

    def __len__(self):
        len1 = len(self.index2word.keys())
        len2 = len(self.word2index.keys())
        len3 = len(self.adjacency_list.keys())
        if len1!=len2 and len2!=len3 and len1!=len3:
            raise ValueError('Expected all three of the adjacency list, word to index map and index to word map to have the same len, got {l3} for adjacency list, {l2} for word to index map and {l1} for index 2 word map')
        return len1
