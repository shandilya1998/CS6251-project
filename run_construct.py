import os
os.chdir('data/')
os.listdir()
from data.graph import df2, graph
#from data.preporcess import prepocess

m_graph = graph(df2).m_construct_graph()
#graph = graph(df2).construct_graph()
os.chdir('..')
import pickle
file = 'data/m_graph.pickle'
pkl = open(file)
pickle.dump(m_graph, pkl)
pkl.close()

from source.pagerank import test1
print(tests.pagerank)


