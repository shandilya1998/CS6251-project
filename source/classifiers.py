import tensorflow as tf
from constants import *

class EmotionClassier(tf.keras.Model):
    def __init__(self, 
            layer1_units, 
            dense_units = NUM_EMOTIONS,
            dense_activation = 'tanh',
            layer1_activation = 'tanh',
            layer1_recurrent_activation='sigmoid',
            layer1_merge_mode = 'concat',
        ):
        super(EmotionClassifier, self).__init__()
        self.bilstm = tf.keras.layers.BiDirectional(
            tf.keras.layers.LSTM(
                units = layer1_units,
                activation = layer1_activation,
                recurrent_activation = layer1_activation,
                return_sequences = False,
                return_state = False,
            )
        )
        self.dense = tf.keras.layers.Dense(
            units = dense_units,
            activation = dense_activation
        )

    def call(self, x):
        x = self.bilstm(x)
        x = self.dense(x)
        return x

