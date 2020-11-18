#model 16
#SE -ELMo
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
import os
import time
import numpy as np
from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Layer, InputSpec, Dense, Input, SpatialDropout1D, LSTM, Activation,Lambda, Embedding, Conv2D, GlobalMaxPool1D, add, concatenate,TimeDistributed
from tensorflow.keras import initializers, regularizers, constraints, activations
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adagrad, Adam
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import Constant
import matplotlib.pyplot as plt
import sys

MODELS_DIR = '/content/drive/My Drive/Shandilya/Padhai/internship/source/checkpoints/model16/SE/'
DATA_SET_DIR = '/content/drive/My Drive/Shandilya/Padhai/internship/source/'

class TimestepDropout(Dropout):
    """Word Dropout.
    This version performs the same function as Dropout, however it drops
    entire timesteps (e.g., words embeddings) instead of individual elements (features).
    # Arguments
        rate: float between 0 and 1. Fraction of the timesteps to drop.
    # Input shape
        3D tensor with shape:
        `(samples, timesteps, channels)`
    # Output shape
        Same as input
    # References
        - N/A
    """

    def __init__(self, rate, **kwargs):
        super(TimestepDropout, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape

class Highway(Layer):
    """Highway network, a natural extension of LSTMs to feedforward networks.
    # Arguments
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        transform_activation: Activation function to use
            for the transform unit
            (see [activations](../activations.md)).
            Default: sigmoid (`sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).x
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        transform_initializer: Initializer for the `transform` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        transform_bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
            Default: -2 constant.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        transform_regularizer: Regularizer function applied to
            the `transform` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        transform_bias_regularizer: Regularizer function applied to the transform bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # References
        - [Highway Networks](http://arxiv.org/pdf/1505.00387v2.pdf)
    """

    def __init__(self,
                 activation='softmax',
                 transform_activation='softmax',
                 kernel_initializer='glorot_uniform',
                 transform_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 transform_bias_initializer=-2,
                 kernel_regularizer=None,
                 transform_regularizer=None,
                 bias_regularizer=None,
                 transform_bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.transform_activation = activations.get(transform_activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.transform_initializer = initializers.get(transform_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        if isinstance(transform_bias_initializer, int):
            self.transform_bias_initializer = Constant(value=transform_bias_initializer)
        else:
            self.transform_bias_initializer = initializers.get(transform_bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.transform_regularizer = regularizers.get(transform_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.transform_bias_regularizer = regularizers.get(transform_bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[-1]

        self.W = self.add_weight(shape=(input_dim, input_dim),
                                 name='{}_W'.format(self.name),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.W_transform = self.add_weight(shape=(input_dim, input_dim),
                                           name='{}_W_transform'.format(self.name),
                                           initializer=self.transform_initializer,
                                           regularizer=self.transform_regularizer,
                                           constraint=self.kernel_constraint)

        self.bias = self.add_weight(shape=(input_dim,),
                                 name='{}_bias'.format(self.name),
                                 initializer=self.bias_initializer,
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.bias_transform = self.add_weight(shape=(input_dim,),
                                           name='{}_bias_transform'.format(self.name),
                                           initializer=self.transform_bias_initializer,
                                           regularizer=self.transform_bias_regularizer)

        self.built = True
        super(Highway, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        x_h = self.activation(K.dot(x, self.W) + self.bias)
        x_trans = self.transform_activation(K.dot(x, self.W_transform) + self.bias_transform)
        output = x_h * x_trans + (1 - x_trans) * x
        return output

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'transform_activation': activations.serialize(self.transform_activation),
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'transform_initializer': initializers.serialize(self.transform_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'transform_bias_initializer': initializers.serialize(self.transform_bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'transform_regularizer': regularizers.serialize(self.transform_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'transform_bias_regularizer': regularizers.serialize(self.transform_bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
                  }
        base_config = super(Highway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Camouflage(Layer):
    """Masks a sequence by using a mask value to skip timesteps based on another sequence.
       LSTM and Convolution layers may produce fake tensors for padding timesteps. We need
       to eliminate those tensors by replicating their initial values presented in the second input.
       inputs = Input()
       lstms = LSTM(units=100, return_sequences=True)(inputs)
       padded_lstms = Camouflage()([lstms, inputs])
       ...
    """

    def __init__(self, mask_value=0., **kwargs):
        super(Camouflage, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, inputs):
        boolean_mask = K.any(K.not_equal(inputs[1], self.mask_value),
                             axis=-1, keepdims=True)
        return inputs[0] * K.cast(boolean_mask, K.dtype(inputs[0]))

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(Camouflage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class SampledSoftmax(Layer):
    """Sampled Softmax, a faster way to train a softmax classifier over a huge number of classes.
    # Arguments
        num_classes: number of classes
        num_sampled: number of classes to be sampled at each batch
        tied_to: layer to be tied with (e.g., Embedding layer)
        kwargs:
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # References
        - [Tensorflow code](tf.nn.sampled_softmax_loss)
        - [Sampled SoftMax](https://www.tensorflow.org/extras/candidate_sampling.pdf)
    """
    def __init__(self, num_classes=50000, num_sampled=1000, tied_to=None, **kwargs):
        super(SampledSoftmax, self).__init__(**kwargs)
        self.num_sampled = num_sampled
        self.num_classes = num_classes
        self.tied_to = tied_to
        self.sampled = (self.num_classes != self.num_sampled)

    def build(self, input_shape):
        if self.tied_to is None:
            self.softmax_W = self.add_weight(shape=(self.num_classes, input_shape[0][-1]), name='W_soft', initializer='lecun_normal')
        self.softmax_b = self.add_weight(shape=(self.num_classes,), name='b_soft', initializer='zeros')
        self.built = True
        super(SampledSoftmax, self).build(input_shape)

    def call(self, x, mask=None):
        lstm_outputs, next_token_ids = x

        def sampled_softmax(x):
            lstm_outputs_batch, next_token_ids_batch = x
            batch_losses = tf.nn.sampled_softmax_loss(
                self.softmax_W if self.tied_to is None else self.tied_to.weights[0], self.softmax_b,
                next_token_ids_batch, lstm_outputs_batch,
                num_classes=self.num_classes,
                num_sampled=self.num_sampled)
            batch_losses = tf.reduce_mean(batch_losses)
            return [batch_losses, batch_losses]

        def softmax(x):
            lstm_outputs_batch, next_token_ids_batch = x
            logits = tf.matmul(lstm_outputs_batch,
                               tf.transpose(self.softmax_W) if self.tied_to is None else tf.transpose(self.tied_to.weights[0]))
            logits = tf.nn.bias_add(logits, self.softmax_b)
            batch_predictions = tf.nn.softmax(logits)
            labels_one_hot = tf.one_hot(tf.cast(next_token_ids_batch, dtype=tf.int32), self.num_classes)
            batch_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
            return [batch_losses, batch_predictions]

        losses, predictions = tf.map_fn(sampled_softmax if self.sampled else softmax, [lstm_outputs, next_token_ids])
        self.add_loss(0.5 * tf.reduce_mean(losses[0]))
        return lstm_outputs if self.sampled else predictions

    def compute_output_shape(self, input_shape):
        return input_shape[0] if self.sampled else (input_shape[0][0], input_shape[0][1], self.num_classes)

#from data import MODELS_DIR
#from .custom_layers import TimestepDropout, Camouflage, Highway, SampledSoftmax


class ELMo(object):
    def __init__(self, parameters, trainable = True):
        self._model = None
        self._elmo_model = None
        self.parameters = parameters
        #self.compile_elmo()
        self.loss_history = []
        self.last_loss = 0.0
        self.val_loss_history = []
        self.file = sys.stdout
        self.trainable = trainable
        #self.optimizer = Adagrad(lr=self.parameters['lr'], clipvalue=self.parameters['clip_value'])

    def __del__(self):
        K.clear_session()
        del self._model

    def char_level_token_encoder(self):
        charset_size = self.parameters['charset_size']
        char_embedding_size = self.parameters['char_embedding_size']
        token_embedding_size = self.parameters['hidden_units_size']
        n_highway_layers = self.parameters['n_highway_layers']
        filters = self.parameters['cnn_filters']
        token_maxlen = self.parameters['token_maxlen']+2

        # Input Layer, word characters (samples, words, character_indices)
        inputs = Input(shape=(None, token_maxlen,), dtype='int32')
        # Embed characters (samples, words, characters, character embedding)
        embeds = Embedding(input_dim=charset_size, output_dim=char_embedding_size, trainable = self.trainable)(inputs)
        token_embeds = []
        # Apply multi-filter 2D convolutions + 1D MaxPooling + tanh
        for (window_size, filters_size) in filters:
            convs = Conv2D(filters=filters_size, kernel_size=[window_size, char_embedding_size], strides=(1, 1),
                           padding="same", trainable = self.trainable)(embeds)
            convs = TimeDistributed(GlobalMaxPool1D(), trainable = self.trainable)(convs)
            convs = Activation('tanh', trainable = self.trainable)(convs)
            convs = Camouflage(mask_value=0, trainable = self.trainable)(inputs=[convs, inputs])
            token_embeds.append(convs)
        token_embeds = concatenate(token_embeds)
        # Apply highways networks
        for i in range(n_highway_layers):
            token_embeds = TimeDistributed(Highway(), trainable = self.trainable)(token_embeds)
            token_embeds = Camouflage(mask_value=0, trainable = self.trainable)(inputs=[token_embeds, inputs])
        # Project to token embedding dimensionality
        token_embeds = TimeDistributed(Dense(units=token_embedding_size, activation='linear'), trainable = self.trainable)(token_embeds)
        token_embeds = Camouflage(mask_value=0, trainable = self.trainable)(inputs=[token_embeds, inputs])

        token_encoder = Model(inputs=inputs, outputs=token_embeds, name='token_encoding')
        return token_encoder
    
    def compile_elmo(self,train_data, val_data, print_summary=False, start = 0, epochs = 5):
        """
        Compiles a Language Model RNN based on the given parameters
        """

        if self.parameters['token_encoding'] == 'word':
            # Train word embeddings from scratch
            word_inputs = Input(shape=(None,), name='word_indices', dtype='int32')
            embeddings = Embedding(self.parameters['vocab_size'], self.parameters['hidden_units_size'], trainable = self.trainable, name='token_encoding')
            inputs = embeddings(word_inputs)

            # Token embeddings for Input
            drop_inputs = SpatialDropout1D(self.parameters['dropout_rate'], trainable = self.trainable)(inputs)
            lstm_inputs = TimestepDropout(self.parameters['word_dropout_rate'], trainable = self.trainable)(drop_inputs)

            # Pass outputs as inputs to apply sampled softmax
            next_ids = Input(shape=(None, 1), name='next_ids', dtype='float32')
            previous_ids = Input(shape=(None, 1), name='previous_ids', dtype='float32')
        elif self.parameters['token_encoding'] == 'char':
            # Train character-level representation
            word_inputs = Input(shape=(None, self.parameters['token_maxlen']+2,), dtype='int32', name='char_indices')
            inputs = self.char_level_token_encoder()(word_inputs)

            # Token embeddings for Input
            drop_inputs = SpatialDropout1D(self.parameters['dropout_rate'], trainable = self.trainable)(inputs)
            lstm_inputs = TimestepDropout(self.parameters['word_dropout_rate'], trainable = self.trainable)(drop_inputs)

            # Pass outputs as inputs to apply sampled softmax
            next_ids = Input(shape=(None, 1), name='next_ids', dtype='float32')
            previous_ids = Input(shape=(None, 1), name='previous_ids', dtype='float32')

        # Reversed input for backward LSTMs
        re_lstm_inputs = Lambda(function=ELMo.reverse)(lstm_inputs)
        mask = Lambda(function=ELMo.reverse)(drop_inputs)

        # Forward LSTMs
        for i in range(self.parameters['n_lstm_layers']):
            lstm = LSTM(units=self.parameters['lstm_units_size'], return_sequences=True, trainable = self.trainable,
                                 kernel_constraint=MinMaxNorm(-1*self.parameters['cell_clip'],
                                                              self.parameters['cell_clip']),
                                 recurrent_constraint=MinMaxNorm(-1*self.parameters['cell_clip'],
                                                                 self.parameters['cell_clip']))(lstm_inputs)#,

                           
            lstm = Camouflage(mask_value=0, trainable = self.trainable)(inputs=[lstm, drop_inputs])
            # Projection to hidden_units_size
            proj = TimeDistributed(Dense(self.parameters['hidden_units_size'], activation= 'relu', trainable = self.trainable))(lstm)#'linear',)
                            
            # Merge Bi-LSTMs feature vectors with the previous ones
            lstm_inputs = add([proj, lstm_inputs], name='f_block_{}'.format(i + 1))
            # Apply variational drop-out between BI-LSTM layers
            lstm_inputs = SpatialDropout1D(self.parameters['dropout_rate'], trainable = self.trainable)(lstm_inputs)

        # Backward LSTMs
        for i in range(self.parameters['n_lstm_layers']):
            re_lstm = LSTM(units=self.parameters['lstm_units_size'], return_sequences=True, trainable = self.trainable,
                                 kernel_constraint=MinMaxNorm(-1*self.parameters['cell_clip'],
                                                              self.parameters['cell_clip']),
                                 recurrent_constraint=MinMaxNorm(-1*self.parameters['cell_clip'],
                                                                 self.parameters['cell_clip']))(re_lstm_inputs)#,
                            

            re_lstm = Camouflage(mask_value=0, trainable = self.trainable)(inputs=[re_lstm, mask])
            # Projection to hidden_units_size
            re_proj = TimeDistributed(Dense(self.parameters['hidden_units_size'], activation= 'relu'), trainable = self.trainable)(re_lstm)#'linear',)
                                          
            # Merge Bi-LSTMs feature vectors with the previous ones
            re_lstm_inputs = add([re_proj, re_lstm_inputs], name='b_block_{}'.format(i + 1))
            # Apply variational drop-out between BI-LSTM layers
            re_lstm_inputs = SpatialDropout1D(self.parameters['dropout_rate'], trainable = self.trainable)(re_lstm_inputs)

        # Reverse backward LSTMs' outputs = Make it forward again
        re_lstm_inputs = Lambda(function=ELMo.reverse, name="reverse")(re_lstm_inputs)

        # Project to Vocabulary with Sampled Softmax
        sampled_softmax = SampledSoftmax(num_classes=self.parameters['vocab_size'], trainable = self.trainable,
                                         num_sampled=int(self.parameters['num_sampled']),
                                         tied_to=embeddings if self.parameters['weight_tying']
                                         and self.parameters['token_encoding'] == 'word' else None)
        outputs = sampled_softmax([lstm_inputs, next_ids])
        re_outputs = sampled_softmax([re_lstm_inputs, previous_ids])

        self._model = Model(inputs=[word_inputs, next_ids, previous_ids],
                            outputs=[outputs, re_outputs])
        self.optimizer = Adam(lr=self.parameters['lr'])#, clipvalue=self.parameters['clip_value'])
        
        self._model.compile(self.optimizer,
                            loss=None)
        if print_summary:
            self._model.summary()
        
    def train(self, train_data, val_data):

        # Add callbacks (early stopping, model checkpoint)
        #train_data = tf.data.Dataset.from_generator(train_data, output_types=())
        weights_file = os.path.join(MODELS_DIR, "elmo_best_weights.ckpt")
        save_best_model = ModelCheckpoint(filepath=weights_file, monitor='loss', verbose=1,
                                          save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(patience=self.parameters['patience'], restore_best_weights=True, monitor = 'loss')

        t_start = time.time()

        # Fit Model
        history = self._model.fit(train_data,
                        #validation_data = val_data,
                        epochs=self.parameters['epochs'],
                        workers=self.parameters['n_threads'],
                        use_multiprocessing=True
                        if self.parameters['multi_processing'] else False,
                        callbacks=[save_best_model, early_stopping])

        print('Training took {0} sec'.format(str(time.time() - t_start)))
        return history

    def train_gt(self, train_data, val_data, epochs = 5, start = 0):
        def train_step(x_true, y_true, epoch, batch):
            loss = None
            with tf.GradientTape() as tape:
                #print(model_train.layers)
                logits = self._model(x_true)
                loss = sum(self._model.losses)
                #if loss.numpy()<300:
                self.loss_history.append(loss.numpy())
                #else:
                #    self.loss_history.append(200.0)
                if batch%375==0:
                    tqdm.write('loss : ' + str(loss.numpy()), file = self.file)
            
            
            grads = tape.gradient(loss, self._model.trainable_variables)
            if batch%750==0:
                tqdm.write('max grad:', file = self.file)
                print([tf.math.reduce_max(grad).numpy() for grad in grads])
                tqdm.write('min grad:', file = self.file)
                print([tf.math.reduce_min(grad).numpy() for grad in grads])
            self.optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        
        for epoch in range(start, start+epochs):
            tqdm.write('epoch: '+str(epoch), file = self.file)
            for (batch, (x_true, y_true)) in tqdm(enumerate(train_data)):
                train_step(x_true, y_true, epoch, batch)
                
            #status.assert_consumed()  
            if epoch % self.parameters['patience'] == 0 or epoch == start:
                self.last_loss = self.loss_history[-1]

            if epoch != start:
                if self.last_loss>self.loss_history[-1]:
                    tqdm.write('saving model', file = self.file)
                    self._model.save_weights(os.path.join(MODELS_DIR,'checkpoints.ckpt'))
            if self.last_loss<self.loss_history[-1]:
                break
            else:
                tqdm.write('saving model', file = self.file)
                self.last_loss = self.loss_history[-1]
                self._model.save_weights(os.path.join(MODELS_DIR,'checkpoints.ckpt'))

            plt.figure(figsize = (16,16), num = epoch)
            plt.subplot(111)
            plt.plot(self.loss_history)
            plt.title('loss')
            plt.savefig(os.path.join(MODELS_DIR, 'loss_'+str(epoch)+'.png'))
            val_loss = 0.0
            val = 0
            for (val, (x_true, y_true)) in tqdm(enumerate(val_data)):
                logits = self._model(x_true)
                val_loss+=sum(self._model.losses)
            self.val_loss_history.append(val_loss)
        
        plt.figure(figsize = (16,16), num = epoch)
        plt.subplot(111)
        plt.plot(self.val_loss_history)
        plt.title('val_loss')
        plt.savefig(os.path.join(MODELS_DIR, 'val_loss.png'))

    def evaluate(self, test_data):

        def unpad(x, y_true, y_pred):
            y_true_unpad = []
            y_pred_unpad = []
            for i, x_i in enumerate(x):
                for j, x_ij in enumerate(x_i):
                    if x_ij == 0:
                        y_true_unpad.append(y_true[i][:j])
                        y_pred_unpad.append(y_pred[i][:j])
                        break
            return np.asarray(y_true_unpad), np.asarray(y_pred_unpad)

        # Generate samples
        x, y_true_forward, y_true_backward = [], [], []
        for i in range(len(test_data)):
            test_batch = test_data[i][0]
            x.extend(test_batch[0])
            y_true_forward.extend(test_batch[1])
            y_true_backward.extend(test_batch[2])
        x = np.asarray(x)
        y_true_forward = np.asarray(y_true_forward)
        y_true_backward = np.asarray(y_true_backward)

        # Predict outputs
        y_pred_forward, y_pred_backward = self._model.predict([x, y_true_forward, y_true_backward])

        # Unpad sequences
        y_true_forward, y_pred_forward = unpad(x, y_true_forward, y_pred_forward)
        y_true_backward, y_pred_backward = unpad(x, y_true_backward, y_pred_backward)

        # Compute and print perplexity
        print('Forward Langauge Model Perplexity: {}'.format(ELMo.perplexity(y_pred_forward, y_true_forward)))
        print('Backward Langauge Model Perplexity: {}'.format(ELMo.perplexity(y_pred_backward, y_true_backward)))

    def wrap_multi_elmo_encoder(self, print_summary=False, save=False):
        """
        Wrap ELMo meta-model encoder, which returns an array of the 3 intermediate ELMo outputs
        :param print_summary: print a summary of the new architecture
        :param save: persist model
        :return: None
        """

        elmo_embeddings = list()
        elmo_embeddings.append(concatenate([self._model.get_layer('token_encoding').output, self._model.get_layer('token_encoding').output],
                                           name='elmo_embeddings_level_0'))
        for i in range(self.parameters['n_lstm_layers']):
            elmo_embeddings.append(concatenate([self._model.get_layer('f_block_{}'.format(i + 1)).output,
                                                Lambda(function=ELMo.reverse)
                                                (self._model.get_layer('b_block_{}'.format(i + 1)).output)],
                                               name='elmo_embeddings_level_{}'.format(i + 1)))

        camos = list()
        for i, elmo_embedding in enumerate(elmo_embeddings):
            camos.append(Camouflage(mask_value=0.0, name='camo_elmo_embeddings_level_{}'.format(i + 1))([elmo_embedding,
                                                                                                         self._model.get_layer(
                                                                                                             'token_encoding').output]))

        self._elmo_model = Model(inputs=[self._model.get_layer('word_indices').input], outputs=camos)

        if print_summary:
            self._elmo_model.summary()

        if save:
            self._elmo_model.save(os.path.join(MODELS_DIR, 'elmo_best_encoder_weights.ckpt'))
            print('ELMo Encoder saved successfully')

    def save(self, sampled_softmax=True):
        """
        Persist model in disk
        :param sampled_softmax: reload model using the full softmax function
        :return: None
        """
        if not sampled_softmax:
            self.parameters['num_sampled'] = self.parameters['vocab_size']
        self.compile_elmo()
        self._model.load_weights(os.path.join(MODELS_DIR, 'elmo_best_weights.ckpt'))
        self._model.save(os.path.join(MODELS_DIR, 'elmo_best_weights.ckpt'))
        print('ELMo Language Model saved successfully')

    def load(self):
        self._model.load_weights(os.path.join(MODELS_DIR, 'checkpoints.ckpt'))

    def load_elmo_encoder(self):
        self._elmo_model = load_model(os.path.join(MODELS_DIR, 'elmo_best_encoder_weights.ckpt'),
                                      custom_objects={'TimestepDropout': TimestepDropout,
                                                      'Camouflage': Camouflage})

    def get_outputs(self, test_data, output_type='word', state='last'):
        """
       Wrap ELMo meta-model encoder, which returns an array of the 3 intermediate ELMo outputs
       :param test_data: data generator
       :param output_type: "word" for word vectors or "sentence" for sentence vectors
       :param state: 'last' for 2nd LSTMs outputs or 'mean' for mean-pooling over inputs, 1st LSTMs and 2nd LSTMs
       :return: None
       """
        # Generate samples
        x = []
        for i in range(len(test_data)):
            test_batch = test_data[i][0]
            x.extend(test_batch[0])

        preds = np.asarray(self._elmo_model.predict(np.asarray(x)))
        if state == 'last':
            elmo_vectors = preds[-1]
        else:
            elmo_vectors = np.mean(preds, axis=0)

        if output_type == 'words':
            return elmo_vectors
        else:
            return np.mean(elmo_vectors, axis=1)

    @staticmethod
    def reverse(inputs, axes=1):
        return K.reverse(inputs, axes=axes)

    @staticmethod
    def perplexity(y_pred, y_true):

        cross_entropies = []
        for y_pred_seq, y_true_seq in zip(y_pred, y_true):
            # Reshape targets to one-hot vectors
            y_true_seq = to_categorical(y_true_seq, y_pred_seq.shape[-1])
            # Compute cross_entropy for sentence words
            cross_entropy = K.categorical_crossentropy(K.tf.convert_to_tensor(y_true_seq, dtype=K.tf.float32),
                                                       K.tf.convert_to_tensor(y_pred_seq, dtype=K.tf.float32))
            cross_entropies.extend(cross_entropy.eval(session=K.get_session()))

        # Compute mean cross_entropy and perplexity
        cross_entropy = np.mean(np.asarray(cross_entropies), axis=-1)

        return pow(2.0, cross_entropy)


class LMDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(self.iterations)

    def __init__(self, corpus, vocab, seed=0, sentence_maxlen=100, token_maxlen=50, batch_size=32, shuffle=True, token_encoding='word', iterations = 2000):
        """Compiles a Language Model RNN based on the given parameters
        :param corpus: filename of corpus
        :param vocab: filename of vocabulary
        :param sentence_maxlen: max size of sentence
        :param token_maxlen: max size of token in characters
        :param batch_size: number of steps at each batch
        :param shuffle: True if shuffle at the end of each epoch
        :param token_encoding: Encoding of token, either 'word' index or 'char' indices
        :return: Nothing
        """
        self.iterations = iterations
        self.corpus = corpus
        self.vocab = {}
        for line in open(vocab, 'r').readlines():
            try:
                line = line[:-1].split(' ')
                self.vocab[line[0]] = int(line[1])
            except:
                pass
        #print(len(self.vocab.keys()))
        temp1 = self.vocab['<pad>']
        temp2 = None
        for word in self.vocab.keys():
            if self.vocab[word]==0:
                temp2 = word
        self.vocab['<pad>'] = 0
        self.vocab[temp2] = temp1
        self.sent_ids = corpus
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sentence_maxlen = sentence_maxlen
        self.token_maxlen = token_maxlen
        self.token_encoding = token_encoding
        self.seed = seed
        with open(self.corpus) as fp:
            self.indices = np.arange(len(fp.readlines()))
            newlines = [index for index in range(0, len(self.indices), 2)]
            self.indices = np.delete(self.indices, newlines)
    
    def __iter__(self):
        for i in range(self.seed, self.seed+self.iterations):
            yield self.__getitem__(i)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Read sample sequences
        word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)
        if self.token_encoding == 'char':
            word_char_indices_batch = np.full((len(batch_indices), self.sentence_maxlen, self.token_maxlen+2), 260, dtype=np.int32)

        for i, batch_id in enumerate(batch_indices):
            # Read sentence (sample)
            word_indices_batch[i] = self.get_token_indices(sent_id=batch_id)
            if self.token_encoding == 'char':
                word_char_indices_batch[i] = self.get_token_char_indices(sent_id=batch_id)

        # Build forward targets
        for_word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)

        padding = np.zeros((1,), dtype=np.int32)

        for i, word_seq in enumerate(word_indices_batch ):
            for_word_indices_batch[i] = np.concatenate((word_seq[1:], padding), axis=0)

        for_word_indices_batch = for_word_indices_batch[:, :, np.newaxis]

        # Build backward targets
        back_word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)

        for i, word_seq in enumerate(word_indices_batch):
            back_word_indices_batch[i] = np.concatenate((padding, word_seq[:-1]), axis=0)

        back_word_indices_batch = back_word_indices_batch[:, :, np.newaxis]

        return (word_indices_batch if self.token_encoding == 'word' else word_char_indices_batch, for_word_indices_batch, back_word_indices_batch), None

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_token_indices(self, sent_id: int):
        with open(self.corpus) as fp:
            for i, line in enumerate(fp):
                if i == sent_id:
                    token_ids = np.zeros((self.sentence_maxlen,), dtype=np.int32)
                    # Add begin of sentence index
                    #token_ids[0] = self.vocab['<bos>']
                    for j, token in enumerate(line[:-1].split()):
                        if token.lower() in self.vocab:
                            token_ids[j] = self.vocab[token.lower()]
                        else:
                            token_ids[j] = self.vocab['<unk>']
                    # Add end of sentence index
                    #if token_ids[1]:
                    #    token_ids[j + 2] = self.vocab['<eos>']
                    return token_ids

    def get_token_char_indices(self, sent_id: int):
        def convert_token_to_char_ids(token, token_maxlen):
            bos_char = 256  # <begin sentence>
            eos_char = 257  # <end sentence>
            bow_char = 258  # <begin word>
            eow_char = 259  # <end word>
            pad_char = 260  # <pad char>
            unk_char = 261  # <unk char>
            char_indices = np.full([token_maxlen+2], pad_char, dtype=np.int32)
            # Encode word to UTF-8 encoding
            word_encoded = token.encode('utf-8', 'ignore')
            # Set characters encodings
            # Add begin of word char index
            char_indices[0] = bow_char
            if token == '<bos>':
                char_indices[1] = bos_char
                k = 1
            elif token == '<eos>':
                char_indices[1] = eos_char
                k = 1
            elif token == '<unk>':
                char_indices[1] = unk_char
                k = 1
            else:
                # Add word char indices
                for k, chr_id in enumerate(word_encoded, start=1):
                    char_indices[k] = chr_id + 1
            # Add end of word char index
            char_indices[k + 1] = eow_char
            return char_indices

        with open(self.corpus) as fp:
            for i, line in enumerate(fp):
                if i == sent_id:
                    token_ids = np.zeros((self.sentence_maxlen, self.token_maxlen+2), dtype=np.int32)
                    # Add begin of sentence char indices
                    #token_ids[0] = convert_token_to_char_ids('<bos>', self.token_maxlen)
                    # Add tokens' char indices
                    for j, token in enumerate(line[:-1].split()):
                        token_ids[j] = convert_token_to_char_ids(token, self.token_maxlen)
                    # Add end of sentence char indices
                    #if token_ids[1]:
                    #    token_ids[j + 2] = convert_token_to_char_ids('<eos>', self.token_maxlen)
        return token_ids
