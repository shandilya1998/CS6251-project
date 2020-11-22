from models import *
from constants import *
import tensorflow as tf
from dataloader import DataLoader
from tqdm import tqdm

def train():
    sent_enc_dense_units = 50
    sent_enc_layer1_units = 50
    sent_enc_layer2_units = 100
    pos_enc_dense_units = 40
    pos_enc_layer1_units = 50
    cl_layer1_units = 35

    dataloader = Dataloader(
        batch_size = BATCH_SIZE,
        train = True,
        train_test_split = 0.9
    )

    model = get_model(
        sent_enc_dense_units,
        sent_enc_layer1_units,
        sent_enc_layer2_units,
        pos_enc_dense_units,
        pos_enc_layer1_units,
        cl_layer1_units,
        num_words = dataloader.generator.num_words
    ) 
    print(model.summary())

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(sent, pos, y): 
        with tf.GradientTape() as tape:
            out = model(sent, pos)
            loss = loss_object(y, out)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(y, out)
    
    @tf.function
    def test_step(sent, pos, y):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        out = model(sent, pos, training=False)
        loss = loss_object(y, out)

        test_loss(loss)
        test_accuracy(y, out)

    train_loss_history = []
    test_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []
    
    for epoch in range(NUM_EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for x, y in tqdm(dataloader.generator):
            train_step(x[0], x[1], y)

        dataloader.toggle(train = False)

        for x, y in dataloader.generator:
            test_step(x[0], x[1], y)

        dataloader.toggle(train = True)

        train_loss_history.append(train_loss.result())
        test_loss_history.append(test_loss.resut())
        train_accuracy_history.append(train_accuracy.result()*100)
        test_accuracy_history.append(test_accuracy.result()*100)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
    return train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history
        
train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history = train()
     
train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history = train() 
pkl = open('out/train_loss_history.pickle', 'wb')
pickle.dump(train_loss_history, pkl)
pkl.close()
pkl = open('out/test_loss_history.pickle', 'wb')
pickle.dump(test_loss_history, pkl)
pkl.close()
pkl = open('out/train_accuracy_history.pickle', 'wb')
pickle.dump(train_accuracy_history, pkl)
pkl.close()
pkl = open('out/test_accuracy_history.pickle', 'wb')
pickle.dump(test_accuracy_history, pkl)
pkl.close()
