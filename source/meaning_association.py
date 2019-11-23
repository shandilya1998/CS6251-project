import pandas as pd
import networkx as nx
import numpy as np 
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

file = 'adjacency_matrix.pickle'
pkl = open(file, 'rb')
G = pickle.load(pkl)
pkl.close()

class meaning_association():
    def __init__(self, G):
        self.G = G
        self.wordlist = list(self.G.nodes)
        # Comment the next 5 lines if the pickle is made already, uncomment lines 6 to 8 after this
        self.m_G = self.compute_neighbor_association()
        file = 'graph_meaning_association1.pickle'
        pkl = open(file, 'wb')
        pickle.dump(self.m_G, pkl)
        pkl.close()
        #pkl = open(file, 'rb')
        #self.m_g = pickle.load(pkl)
        #pkl.close()

    def set_meaning_association(self, w, def_w, G):
        """
            This methods takes the word w, its definition def_w and the networkx graph G
            Computes the meaning association between two graphs 
        """
        # n_simple_paths is the number of simple paths between w and def_w
        shortest_path_w_def_w = nx.shortest_path_length(G, w, def_w)
        try: 
            shortest_path_def_w_w = nx.shortest_path_length(G, def_w, w)
            try:
                G[w][def_w]['meaning_association'] = float(2/(shortest_path_def_w_w+shortest_path_w_def_w))
            except ZeroDivisionError:
                G[w][def_w]['meaning_association'] = 1
        except nx.exception.NetworkXNoPath:
            G[w][def_w]['meaning_association'] = 0
        return G

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

    def compute_neighbor_association(self, G = None):
        """
            This method takes a networkx graph as input
            Computes the meaning association socre for each edge 
            Returns : networkx graph
        """
        if not G:
            G = self.G
        for word in tqdm(self.wordlist):
            for def_w in G[word]:
                G = self.set_meaning_association(word, def_w, G)
        return G

    def compute_global_association(self, w1, w2):
        paths = list(node_disjoint_paths(self.m_G, w1, w2))
        num = len(paths)
        val = 0
        for path in paths:
            i = 0
            rel = 1.0
            while(True):
                if i == 0:
                    i+=1
                    continue
                elif i == len(path)-1:
                    break
                else:
                    rel = rel*G[path[i-1]][path[i]]
                    i+=1
            val+=rel
        val = val/num
        return val

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


ob = meaning_association(G)


    


