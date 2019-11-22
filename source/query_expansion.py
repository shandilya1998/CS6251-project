import networx as nx
import pickle
file = '../data/corpus.pickle'
pkl = open(file, 'rb')
corpus = pickle.load(pkl)
pkl.close()

file = '../data/graph_meaning_associated1.pickle'
pkl = open(file, 'rb')
G_m = pickle.load(pkl)
pkl.close()

from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')
import string

class query:
    def __init__(self, Q ):
        self.Q = Q
        self.V = G_m.nodes
        self.source = ''

    def create_one_hot_encoded(self):
        q = pd.Series(index = self.V)
        for word in self.bow():
            q.loc[word] == 1
        q = q.fillna(0)
        return q

    def m_create_encoded(self):
        q = pd.Series(index = self.V)
        for word in self.bow():
            G_word = self.get_subgraph(word)
            q = self.populate(G_word)
        q = fillna(0.0)
        return q


    def progressive_widening_search(G, source, value, condition, initial_width=1):
        """Progressive widening beam search to find a node.

        The progressive widening beam search involves a repeated beam
        search, starting with a small beam width then extending to
        progressively larger beam widths if the target node is not
        found. This implementation simply returns the first node found that
        matches the termination condition.

        `G` is a NetworkX graph.

        `source` is a node in the graph. The search for the node of interest
        begins here and extends only to those nodes in the (weakly)
        connected component of this node.

        `value` is a function that returns a real number indicating how good
        a potential neighbor node is when deciding which neighbor nodes to
        enqueue in the breadth-first search. Only the best nodes within the
        current beam width will be enqueued at each step.

        `condition` is the termination condition for the search. This is a
        function that takes a node as input and return a Boolean indicating
        whether the node is the target. If no node matches the termination
        condition, this function raises :exc:`NodeNotFound`.

        `initial_width` is the starting beam width for the beam search (the
        default is one). If no node matching the `condition` is found with
        this beam width, the beam search is restarted from the `source` node
        with a beam width that is twice as large (so the beam width
        increases exponentially). The search terminates after the beam width
        exceeds the number of nodes in the graph.
        """
        # Check for the special case in which the source node satisfies the
        # termination condition.
        self.source = source
        if self.condition(source):
            return source
        # The largest possible value of `i` in this range yields a width at
        # least the number of nodes in the graph, so the final invocation of
        # `bfs_beam_edges` is equivalent to a plain old breadth-first
        # search. Therefore, all nodes will eventually be visited.
        #
        log_m = math.ceil(math.log(len(G_m), 2))
        for i in range(log_m):
            width = initial_width * pow(2, i)
            # Since we are always starting from the same source node, this
            # search may visit the same nodes many times (depending on the
            # implementation of the `value` function).
            for u, v in nx.bfs_beam_edges(G_m, source, value, width):
                self.source = u
                if self.condition(v):
                    break
        # At this point, since all nodes have been visited, we know that
        # none of the nodes satisfied the termination condition.
        raise nx.NodeNotFound("no node satisfied the termination condition")

    def condition(self, node):
        """
            this method takes the node : str to test for condition as input
            self.source is also a parameter set before calling the method
        """
        if G_m[self.source][node]['meaning_association'] < 0.3:
            return False
        else:
            return True

    def bow(self):
        self.Q = npl(self.Q)
        bow_ = []
        for word in self.Q:
            for token in definition:
                if token.text in string.punctuation or token.text == '\'s' or token.text == '':
                    continue
                if word not in self.V:
                    continue
                else:
                    yield token.lemma_

 






