import networkx as nx 
import os
import numpy as np 
import pandas as pd
import pickle
from tqdm import tqdm
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')
from meaning_association import meaning_association
from query_expansion import query

file = '../data/all_words.pickle'
pkl = open(file, 'rb')
wordlist = pickle.load(pkl)
pkl.close()

def pagerank(G, alpha=0.85, personalization=None,
             max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
             dangling=None):
    """Return the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float, optional
      Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key for every graph node and nonzero personalization value for each node.
      By default, a uniform distribution is used.

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
      Starting value of PageRank iteration for each node.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified). This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
       Dictionary of nodes with PageRank as value

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The PageRank algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs by converting each edge in the
    directed graph to two edges.


    """
    if len(G) == 0:
        return {}

    #if not G.is_directed():
    #    D = G.to_directed()
    #else:
    #    D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(G)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    # Initialization of the intial pagerank values
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:

        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError('Personalization dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:

        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v/s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in tqdm(range(max_iter)):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:

            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]

        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N*tol:
            return x
    raise NetworkXError('pagerank: power iteration failed to converge '
                        'in %d iterations.' % max_iter)
class tests:
    def __init__(self, G):
        self.G = G

    def to_graph(self, G_pandas):
        G = nx.from_pandas_adjacency(G, nx.DiGraph)
        return G

    def pagerank(self):
        return pagerank(self.G)

    def __str__(self):
        return "This is the class of all tests performed on the gaph"

<<<<<<< HEAD
file = 'm_graph.pickle'
=======
file = 'adjacency_matrix.pickle'
>>>>>>> 2c78ec15aa36a35e398cdc955b1592c255caf017
pkl = open(file, 'rb')
G = pickle.load(pkl)
pkl.close()

test1 = tests(G)
pagerank_ =  test1.pagerank()
print(len(pagerank_))
file_ = 'pagerank.pickle'
pkl = open(file_, 'wb')
pickle.dump(pagerank_, pkl)
pkl.close()
#print('pagerank with personalization with words in query set high and other set low')
q = 'Programming gives some people joy while others it does not' #test1
#q = nlp(q)
#q = [token.lemma_ for token in q if token.lemma_ in wordlist]
q = query(q)
# get sub-graph with neighbor meaning association values
m_q = q.get_sub_graph()
q_pagerank = pagerank(m_q)
file = 'q_pagerank-test1.pickle'
pkl = open(file, 'wb')
pickle.dump(q_pagerank, pkl)
pkl.close()




