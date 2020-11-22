import pickle
import numpy as np
import math
import copy
import tensorflow as tf
from constants import *

import pandas as pd

import spacy
nlp = spacy.load('en_core_web_sm')

class DataLoader(tf.keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        train,
        train_test_split,
        dictionary_encoding = True,
        num_neighbours = 3,
        adj = ADJ_LIST,
        wordlist = WORDLIST,
        index2word = INDEX2WORD,
    ):
        self.batch_size = batch_size
        self.nn = num_neighbours
        self.g = pickle.load(open(adj, 'rb'))
        self.data = list(self.g.keys())
        if train:
            self.data=self.data[:math.floor(train_test_split*len(self.data))]
        else:
            self.data=self.data[math.floor(train_test_split*len(self.data)):]
        self.wordlist = pickle.load(open(wordlist, 'rb'))
        self.index2word = pickle.load(open(index2word, 'rb'))
        self.num_words = len(self.wordlist.keys())
        self.de = dictionary_encoding

    def get_dict_vec(self, index):
        vec = np.zeros(self.num_words)
        value = 1
        vec[index] = 1/value
        queue = []
        queue.extend(self.g[index])
        value = 2
        i = 0
        flag = len(queue)
        while(len(queue)>1):
            index = queue.pop(0)
            vec[index] = 1/value
            queue.extend(self.g[index])
            i += 1
            if flag == i:
                flag = len(queue)
                i = 0
                #print(value)
                value += 1
            if value == self.nn+2:
                break
        return vec/np.sum(vec)

    def shuffle(self):
        self.data = self.data.sample(frac = 1).reset_index(drop=True)

    def __len__(self):
        return math.ceil(len(self.data)/self.batch_size)

    def __getitem__(self, idx):
        indices = range(self.batch_size*idx, self.batch_size*(1+idx))
        x = []
        y = []
        #print(idx)
        #print(indices)
        for i in indices:
            #print(i)
            y.append(self.data[i])
            x.append(self.get_dict_vec(i))
        x = np.array(x, dtype = np.float32)
        y = np.array(y, dtype = np.int32)
        return x, y

class Compression(tf.keras.Model):
    def __init__(self, 
            layer1_units,
            layer1_activation,
            layer2_units,
            layer2_activation
        ):
        super(Compression, self).__init__()
        self.dense1 = tf.keras.layers.Dense(
            units = layer1_units, 
            activation = layer1_activation
        )
        self.dense2 = tf.keras.layers.Dense(
            units = layer2_units, 
            activation = layer2_activation
        )

    def call(self, x):
        h = self.dense1(x)
        y = self.dense2(h)
        return h, y

def get_model(
    layer1_units,
        layer1_activation,
        layer2_units,
        layer2_activation
):
    x = tf.keras.Input(
        shape = (NUM_WORDS,),
        batch_size = BATCH_SIZE
    )
    h, y = Compression(
        layer1_units,
        layer1_activation,
        layer2_units,
        layer2_activation
    )(x)
    return tf.keras.Model(
        inputs = [x],
        outputs = [h, y]
    )

def train():
    layer1_units = 128
    layer1_activation = 'softmax'
    layer2_units = NUM_WORDS
    layer2_activation = 'softmax'

    model = get_model(
        layer1_units,
        layer1_activation,
        layer2_units,
        layer2_activation
    )

    print(model.summary())
    
    dataloader = DataLoader(
        batch_size = BATCH_SIZE,
        train = True,
        train_test_split = 1.0
    )

    loss = tf.keras.metrics.Mean(name='train_loss')
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    loss_object =tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(x, y): 
        with tf.GradientTape() as tape:
            _, out = model(x)
            loss = loss_object(y, out)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss(loss)
        accuracy(y, out)
    
    loss_history = []
    accuracy_history = []
    
    for epoch in range(NUM_EPOCHS_WORD_EM):
        # Reset the metrics at the start of the next epoch
        loss.reset_states()
        accuracy.reset_states()

        for x, y in tqdm(dataloader):
            train_step(x, y)

        loss_history.append(loss.result())
        accuracy_history.append(accuracy.result()*100)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
        )
    return loss_history, accuracy_history

"""
loss_history, accuracy_history = train() 
pkl = open('word_em_loss_history.pickle', 'wb')
pickle.dump(loss_history, pkl)
pkl.close()
pkl = open('word_em_accuracy_history.pickle', 'wb')
pickle.dump(accuracy_history, pkl)
pkl.close()
"""
