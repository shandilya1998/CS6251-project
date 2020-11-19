from models import *
from constants import *
import tensorflow as tf
from dataloader import DataLoader
 
def train():
    dataloader_train = DataLoader(
        batch_size = BATCH_SIZE,
        train = True,
        train_test_split = 0.9
    )

    dataloader_test = DataLoader(
        batch_size = BATCH_SIZE,
        train = False,
        train_test_split = 0.9
    )

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

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    
    for epoch in range(NUM_EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for x, y in dataloader_train:
            train_step(x[0], x[1], y)

        for x, y in dataloader_test:
            test_step(x[0], x[1], y)

        train_loss.append(train_loss.result())
        test_loss.append(test_loss.resut())
        train_accuracy.append(train_accuracy.result()*100)
        test_accuracy.append(test_accuracy.result()*100)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
