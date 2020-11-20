import tensorflow as tf
from constants import *
from bilstm_encoders import BiLSTMWordVector, BiLSTMPOSVector
from classifiers import EmotionClassier

def get_model(
        sent_enc_dense_units,
        sent_enc_layer1_units,
        sent_enc_layer2_units,
        pos_enc_dense_units,
        pos_enc_layer1_units,
        cl_layer1_units,
        num_words = NUM_WORDS, 
        batch_size = BATCH_SIZE
    ):
    pos = tf.keras.Input(
        shape = (25,),
        batch_size = batch_size,
        dtype = tf.dtypes.int32
    )
    sent = tf.keras.Input(
        shape = (25, num_words),
        batch_size = batch_size,
    )
    
    sent_enc = BiLSTMWordVector(
        dense_units = sent_enc_dense_units,
        layer1_units = sent_enc_layer1_units,
        layer2_units = sent_enc_layer2_units,
    )(sent)
    
    pos_enc = BiLSTMPOSVector(
        dense_units = pos_enc_dense_units,
        layer1_units = pos_enc_layer1_units
    )(pos)

    word_em = tf.concat([sent_enc, pos_enc], axis = -1)

    out = EmotionClassifier(
        layer1_units =  cl_layer1_units
    )(word_em)

    return tf.keras.Model(
        inputs = [sent, pos], 
        outputs = out
    )
