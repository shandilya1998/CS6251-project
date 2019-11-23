print('Preparing data')
from data.raw_data import data
print('Creating an adjacency list')
from data.graph import graph
print('Creating a networkx adjacency matrix')
from data.list_to_matrix import g
print('Computing neighbour association matrix for the graph')
from source.meaning_association import ob
print('create corpus for query expansion test')
import source.create_corpus
print('Calculating pagerank score of each word')
from source.pagerank import pagerank
print('Getting strongly connect components in the graph')
from source.scc import scc



