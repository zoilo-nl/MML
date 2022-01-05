import os
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
#os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from Constants import *

tf.debugging.set_log_device_placement(False)

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", gpus)

#tf.config.experimental.set_memory_growth(gpus[0], True)

def limitgpu(maxmem, gpus):
	if gpus:
		# Restrict TensorFlow to only allocate a fraction of GPU memory
		try:
			for gpu in gpus:
				tf.config.experimental.set_virtual_device_configuration(gpu,
						[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=maxmem)])
		except RuntimeError as e:
			# Virtual devices must be set before GPUs have been initialized
			print(e)

#limitgpu(1024+512, gpus)


model = Sequential()
model.add(BatchNormalization(input_shape=(N_BINS,1,), name="Normalization"))
# model.add(LSTM(256, activation=tf.nn.tanh, recurrent_activation=tf.nn.sigmoid, name="LSTM_1", return_sequences=True, dropout=0.1, recurrent_dropout=0))
# model.add(LSTM(256, activation=tf.nn.tanh, recurrent_activation=tf.nn.sigmoid, name="LSTM_2", return_sequences=True, dropout=0.1, recurrent_dropout=0))
# model.add(LSTM(256, activation=tf.nn.tanh, recurrent_activation=tf.nn.sigmoid, name="LSTM_3", return_sequences=True, dropout=0.1, recurrent_dropout=0))
model.add(LSTM(256, activation=tf.nn.tanh, recurrent_activation=tf.nn.sigmoid, name="LSTM_4", return_sequences=False, dropout=0.1, recurrent_dropout=0))
model.add(Dense(N_NOTES, activation=tf.nn.sigmoid, name="Output"))

model.compile(
    loss=BinaryCrossentropy(from_logits=False),
    optimizer=Adam(learning_rate=0.01),
    metrics=['accuracy'])
model.summary()

# Load metadata from csv file and filter out non-piano music files
metadata = pd.read_csv(PATH_METADATA)
metadata.drop(metadata[metadata["ensemble"] != "Solo Piano"].index, inplace = True)


ids = metadata["id"].to_numpy()
x = np.concatenate([np.load(PATH_DATA+str(fname)+".npy").transpose() for fname in ids])
y = np.concatenate([np.load(PATH_LABELS+str(fname)+".npy") for fname in ids])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None, shuffle=False)
"""
train_files = [f for f in os.listdir(PATH_TRAIN_DATA) if f.endswith('.npy')]
test_files = [f for f in os.listdir(PATH_TEST_DATA) if f.endswith('.npy')]
x_train = np.concatenate([np.load(PATH_TRAIN_DATA+fname).transpose() for fname in train_files])
y_train = np.concatenate([np.load(PATH_TRAIN_LABELS+fname) for fname in train_files])
x_test = np.concatenate([np.load(PATH_TEST_DATA+fname).transpose() for fname in test_files])
y_test = np.concatenate([np.load(PATH_TEST_LABELS+fname) for fname in test_files])
"""

checkpoint_path = "LSTM_training/{epoch}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size = 100
epochs = 2

# Create a callback that saves the model's weights every epoch
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch')

# with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
with tf.device("/CPU:0"):
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[cp_callback])

np.save('LSTM_training/history.npy', history.history)
model.save('LSTM_training/model.h5')