import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa, librosa.display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

PATH_METADATA = "musicnet_metadata.csv"

PATH_DATA = "musicnet/data/"
PATH_LABELS = "musicnet/labels/"

PATH_TRAIN_DATA = "musicnet/train_data/"
PATH_TRAIN_LABELS = "musicnet/train_labels/"
PATH_TEST_DATA = "musicnet/test_data/"
PATH_TEST_LABELS = "musicnet/test_labels/"

ORIGINAL_SR = 44100
TARGET_SR = 16000
FMIN = librosa.note_to_hz("A0")
FMIN_MIDI_INDEX = librosa.note_to_midi("A0")
N_NOTES = 88
BINS_PER_NOTE = 1
BINS_PER_OCTAVE = 12 * BINS_PER_NOTE
N_BINS = N_NOTES * BINS_PER_NOTE

WINDOW_LENGTH = 2048
HOP_LENGTH = 512

frac_sr = TARGET_SR / ORIGINAL_SR
sample_indexer = frac_sr / HOP_LENGTH

tf.debugging.set_log_device_placement(False)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = Sequential()
#model.add(layers.BatchNormalization(input_shape=(N_BINS,)))
#model.add(layers.Dense(1024, activation=tf.nn.relu))
model.add(layers.Dense(1024, activation=tf.nn.relu, input_shape=(N_BINS,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation=tf.nn.relu))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(N_NOTES, activation=tf.nn.sigmoid))

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(),
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

checkpoint_path = "DNN_training/{epoch}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size = 100
epochs = 5

# Create a callback that saves the model's weights every epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

model.save('DNN_training/model.h5')