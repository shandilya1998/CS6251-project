import tensorflow as tf
from constants import *
from bilstm_encoders import BiLSTMWordVector, BiLSTMPOSVector
from classifiers import EmotionClassier

def model(
        sent_enc_dense_units,
        sent_enc_layer1_units,
        sent_enc_layer2_units,
        sent_enc_layer3_units,
        pos_enc_dense_units,
        pos_enc_layer1_units,
        cl_layer1_units,
        num_words = NUM_WORDS, 
        batch_size = BATCH_SIZE
    ):
    pos = tf.keras.Input(
        shape = (25,)
        batch_size = batch_size
    )
    sent = tf.keras.Input(
        shape = (25, num_words)
        batch_size = batch_size
    )
    
    sent_enc = BiLSTMWordVector(
        dense_units = sent_enc_dense_units
        layer1_units = sent_enc_layer1_units,
        layer2_units = sent_enc_layer2_units,
        layer3_units = sent_enc_layer3_units
    )(sent)
    
    pos_enc = BiLSTMPOSVector(
        dense_units = pos_enc_dense_units
        layer1_units = pos_enc_layer1_units
    )(pos)

    word_em = tf.concat([sent_enc, pos_enc], axis = -1)

    out = EmotionClassier(
        layer1_units =  cl_layer1_units
    )(word_em)

    return tf.keras.Model(
        inputs = [sent, pos], 
        outputs = out
    ) 
