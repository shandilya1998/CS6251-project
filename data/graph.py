import pandas as pd
from raw_data import data # data is the object of class Data, which loads wordlist.csv into a DataFrame
import os
import matplotlib.pyplot as plt
import copy 
from raw_data import unique_word_keys_file, unique_words_def_file

class graph_v1():
    def __init__(self):
        self.data = data.sample
        # self.unique_word_keys is a tuple with two elements - (list of words, number of words)
        self.unique_words_keys = self.open_pickle_file(unique_word_keys_file)
        # self.unique_words_def_freq is a tuple with two elements - (list of words&freq, number of words)
        self.unique_words_def_freq = self.open_pickle_file(unique_words_def_file)

    def split(self, data):
        data[1] = data[1].apply(str.split, ' ')
        return data

    def open_pickle_file(self, file):
        pkl = open(file, 'rb')
        data = pickle.load(pkl)
        pickle.close()
        return data
   
   def create_adjacency_matrix(self):
       self.ob_ajncy_m = adjacency_matrix(self.data, len(self.intersection))
       self.adjacency_matrix, self.unique_words = self.ob_adjncy_m.construct()

   def intersection(self):
       unique_words = np.intersect1d(self.unique_words_def_freq, unique_words, keys)
       return unique_words

   def create_adjacency_list(self):
       def.ob_adjcncy_lst = adjacency_list(self.data, len(self.intersection))

class adjacency_matrix():
    def __init__(self, data, size, unique_words):
        self.data = data
        self.unique_words = unique_words
        self.adjacency_matrix = np.zeros((size, size))

    def print_adjncy_matrix(self):
        print(self.adjacency_matrix)

    def construct(self):
        self.wordlist = np.array(())
        for i in range(len(sample)):
            if sample[i, 0] not in self.wordlist:
                self.wordlist = np.append(self.wordlist, np.array([sample[i, 0]]))
            for j in range(len(sample[i, 1])):
                if sample[i,1][j] in self.wordlist:
                    self.adjacency_matrix[self.index_wordlist(sample[i, 0]),self.index_wordlist(sample[i, 1][j])]+=1
                else:
                    self.wordlist = np.append(self.wordlist, np.array([sample[i, 1][j]]))
                    self.adjacency_matrix[self.index_wordlist(sample[i, 0]), len(wordlist)-1]+=1
        return self.adjacency_matrix, self.wordlist

    def index_wordlist(self, word):
        return list(self.wordlist).index(word)

class adjacency_list():
    def __init__(self, data, size, unique_words):
        self.data = data
        self.unique_words = unique_words
        self.adjacency_list = np.zeros((size, 2))

    def print_adjncy_matrix(self):
        print(self.adjacency_list)

    def construct(self):
        self.wordlist = np.array(())
        for i in range(len(sample)):
            if wordlist = np.append(self.wordlist, np.array([sample[i, 0]])):
                self.wordlist = np.append(self.wordlist, np.array([sample[i, 0]]))
            for j in range(len(sample[i, 1])):
                if sample[i,1][j] in self.wordlist:
                    self.adjacency_list[self.index_wordlist(sample[i, 0]), 1].append(j)
                else:
                    self.wordlist = np.append(self.wordlist, np.array([sample[i, 1][j]]))
                    self.adjacency_list[self.index_wordlist(sample[i, 0]), 1].append(j)  
        return self.adjacency_list, self.wordlist

    def index_wordlist(self, word):
        return list(self.wordlist).index(word)

class Relationship():
    def __init__(self, data):
        self.data = data

    """
        A recursive function that find finds and prints strongly connected 
        components using DFS traversal u --> v.
        The vertex to be visited next:
        disc[] --> Stores discovery times of visited vertices 
        low[] --> earliest visited vertex (the vertex with minimum
                    discovery time) that can be reached from subtree
                    rooted with current vertex
        st --> To store all the connected ancestors (could be part
                 of SCC)
        stackMember[] --> bit/index array for faster check whether 
                          a node is in stack
    """
    def SCCUtil(self,u, low, disc, stackMember, st):

        # Initialize discovery time and low value
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1
        stackMember[u] = True
        st.append(u)

        # Go through all vertices adjacent to this
        for v in self.graph[u]:

            # If v is not visited yet, then recur for it
            if disc[v] == -1 :

                self.SCCUtil(v, low, disc, stackMember, st)

                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                # Case 1 (per above discussion on Disc and Low value)
                low[u] = min(low[u], low[v])

            elif stackMember[v] == True:

                '''Update low value of 'u' only if 'v' is still in stack
                (i.e. it's a back edge, not cross edge).
                Case 2 (per above discussion on Disc and Low value) '''
                low[u] = min(low[u], disc[v])

        # head node found, pop the stack and print an SCC
        w = -1 #To store stack extracted vertices
        if low[u] == disc[u]:
            while w != u:
                w = st.pop()
                print w,
                stackMember[w] = False

            print""



    #The function to do DFS traversal.
    # It uses recursive SCCUtil()
    def SCC(self):

        # Mark all the vertices as not visited
        # and Initialize parent and visited,
        # and ap(articulation point) arrays
        disc = [-1] * (self.V)
        low = [-1] * (self.V)
        stackMember = [False] * (self.V)
        st =[]


        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in range(self.V):
            if disc[i] == -1:
                self.SCCUtil(i, low, disc, stackMember, st)


