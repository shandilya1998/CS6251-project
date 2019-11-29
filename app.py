import pandas as pd
import networkx as nx
import os 
import pickle
from data.list_to_matrix import convert
from source.meaning_association import meaning_association


class test1:
    def __init__(self, files = 'wordlist/'):
        self.cwd = os.getcwd()
        self.p_files = os.path.join(self.cwd, files)
        self.wordlist = 'wordlist.pickle'
        self.files = [f for_, __, f in os.walk(self.p_files) if f != 'wordlist.pickle']

    def create_meaning_graphs(self, f):
        g = data_(f).construct_graph()
        g = meaning_association(g).m_g
        return g

    def main(self):
        for f in self.files:
            data_f = data_(os.path.join(self.p_files, f))
            data_f.construct_graph()
            g = data_f.g
            m_g = meaning_association(g)

    def queries(self):
        return ['cat licks glass', 'animals lick glass', 'barley rots', 'barley bags', '']
        

class data_:
    def __init__(self, source):
        self.source = source

    def construct_graph(self):
        self.c = convert(self.source)
        self.g = self.c.g
        return self.g
        
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

