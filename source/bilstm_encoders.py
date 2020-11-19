import tensorflow as tf
from constants import *

class BiLSTMWordVector(tf.keras.Model):
    def __init__(self, 
            dense_units,
            layer1_units,
            layer2_units,
            layer3_units,
            dense_activation = 'tanh',
            layer1_activation = 'tanh',
            layer1_recurrent_activation='sigmoid',
            layer1_merge_mode = 'concat',
            layer2_activation = 'tanh',
            layer2_recurrent_activation='sigmoid',
            layer2_merge_mode = 'concat',
            layer3_activation = 'tanh',
            layer3_recurrent_activation='sigmoid',
            layer3_merge_mode = 'concat',
            max_sent_length = MAX_SENT_LENGTH
        ):
        super(BiLSTMWordVector, self).__init__()
        self.max_sent_length = max_sent_length
        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                units = dense_units,
                activation = dense_activation
            )
        )
        self.bilstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units = layer1_units,
                activation = layer1_activation,
                recurrent_activation = layer1_activation,
                return_sequences = True,
                return_state = False,
            ),
            merge_mode = layer1_merge_mode
        )
        self.bilstm2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units = layer2_units,
                activation = layer2_activation,
                recurrent_activation = layer2_activation,
                return_sequences = True,
                return_state = False,
            ),  
            merge_mode = layer2_merge_mode
        )  
        self.bilstm3 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units = layer3_units,
                activation = layer3_activation,
                recurrent_activation = layer3_activation,
                return_sequences = True,
                return_state = False,
            ),  
            merge_mode = layer3_merge_mode
        ) 
    
    def call(self, x):
        """
            x (batch_size, max_sent_length, num_words)
        """
        x = self.dense(x)
        x = self.bilstm1(x)
        x = self.bilstm2(x)
        x = self.bilstm3(x)
        return x

class BiLSTMPOSVector(tf.keras.Model):
    def __init__(self,
            dense_units,
            layer1_units,
            dense_activation = 'tanh',
            layer1_activation = 'tanh',
            layer1_recurrent_activation='sigmoid',
            layer1_merge_mode = 'concat',
            max_sent_length = MAX_SENT_LENGTH,    
            num_word = NUM_WORDS
            num_tags = NUM_TAGS
        ):
        super(BiLSTMPOSVector, self).__init__()
        self.max_sent_length = max_sent_length
        self.num_words = num_words
        self.num_tags = num_tags
        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                units = dense_units,
                activation = dense_activation
            )
        )
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units = layer1_units,
                activation = layer1_activation,
                recurrent_activation = layer1_activation,
                return_sequences = True,
                return_state = False,
            ),
            merge_mode = layer1_merge_mode
        )
   
    def call(self, x):
        """
            x (batch_size, max_sent_length)
        """
        x = tf.one_hot(x, self.num_tag+1)
        x = self.dense(x)
        x = self.bilstm(x)
        return x        
        
#model = BiLSTMWordVector(512, 384, 256, 128)
#model.build((10, 25, 65000))
#print(model.summary())
