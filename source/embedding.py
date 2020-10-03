import pickle
import copy
from graph import Graph
import tensorflow as tf

class Embedding:
    def __init__(self, path = '../data/wordlist.csv'): 
        self.path = path
        self.data = pd.read_csv(self.path, encoding = 'latin1',  header = None, engine = 'c' )
        self.data = self.data.iloc[:, 1:]
        self.data.fillna('')
        self.data.columns = [col-1 for col in self.data.columns]
        self.graph = Graph()
        
    def create_input_vec(self, start, depth):
        visited = [False] * (len(self.graph))
        vec = np.zeros((len(self.graph)))
        d = 0
        count = 0
        num_item = 1
        out = {}
        queue = {}
        queue.append((start, d))
        visited[start] = True
        num_item = len(self.graph[start])
        
        while queue:
            if d == depth:
                break
            item = queue.pop(0)
            out.append(item)
            count += 1
            for i in self.graph[item[0]]:
                if visited[i]==False:
                    queue.append((i, d))
                    visited[i] = True
            if count==num_item:
                d+=1
                count = 0 
                num_item = len(queue)

        for item in out:
            vec[item[0]] = 1/(item[1]+1)

        return vec

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        self.dense1 = tf.keras.layers.Dense(
            units = 2000, 
            activation = 'tanh' 
        )
    
    def call(self, x):
        x = self.dense1(x)
        return x
