import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks, losses, optimizers
from sklearn.model_selection import train_test_split
from Constants import *

tf.debugging.set_log_device_placement(False)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = models.Sequential()
model.add(layers.BatchNormalization(input_shape=(N_BINS,1,), name="Normalization"))
# model.add(layers.LSTM(256, activation=tf.nn.tanh, recurrent_activation=tf.nn.sigmoid, name="LSTM_1", return_sequences=True, dropout=0.1, recurrent_dropout=0))
# model.add(layers.LSTM(256, activation=tf.nn.tanh, recurrent_activation=tf.nn.sigmoid, name="LSTM_2", return_sequences=True, dropout=0.1, recurrent_dropout=0))
# model.add(layers.LSTM(256, activation=tf.nn.tanh, recurrent_activation=tf.nn.sigmoid, name="LSTM_3", return_sequences=True, dropout=0.1, recurrent_dropout=0))
model.add(layers.LSTM(256, activation=tf.nn.tanh, recurrent_activation=tf.nn.sigmoid, name="LSTM_4", return_sequences=False, dropout=0.1, recurrent_dropout=0))
model.add(layers.Dense(N_NOTES, activation=tf.nn.sigmoid, name="Output"))

model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False),
    optimizer=optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy'])
model.summary()

# Load metadata from csv file and filter out non-piano music files
metadata = pd.read_csv(PATH_METADATA)
metadata.drop(metadata[metadata["ensemble"] != "Solo Piano"].index, inplace = True)

ids = metadata["id"].to_numpy()
test_ids = [1759, 2303, 2556]
train_ids = [x for x in ids if x not in test_ids]
"""
seed = 42
x = np.concatenate([np.load(PATH_DATA+str(fname)+".npy").transpose() for fname in ids])
y = np.concatenate([np.load(PATH_LABELS+str(fname)+".npy") for fname in ids])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
"""
x_train = np.concatenate([np.load(PATH_TRAIN_DATA+str(fname)+".npy").transpose() for fname in train_ids])
y_train = np.concatenate([np.load(PATH_TRAIN_LABELS+str(fname)+".npy") for fname in train_ids])
x_test = np.concatenate([np.load(PATH_TEST_DATA+str(fname)+".npy").transpose() for fname in test_ids])
y_test = np.concatenate([np.load(PATH_TEST_LABELS+str(fname)+".npy") for fname in test_ids])

checkpoint_path = "LSTM_training/{epoch}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size = 100
epochs = 5

# Create a callback that saves the model's weights every epoch
cp_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')

with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

np.save('LSTM_training/history.npy', history.history)
model.save('LSTM_training/model.h5')