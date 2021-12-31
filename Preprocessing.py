import numpy as np
import pandas as pd
import librosa
from Constants import *

# Load wav files, downsample the contents, compute their CQT spectrum, and save it with the same name to the same location as a binary file
def process_wav(id):
    #path_train = PATH_TRAIN_DATA + str(id)
    #path_test = PATH_TEST_DATA + str(id)
    #path = path_train if os.path.isfile(path_train + ".wav") else path_test
    path = PATH_DATA + str(id)

    data_downsampled, sr = librosa.load(path + ".wav", sr=TARGET_SR, res_type='kaiser_best')
    data_cqt = np.abs(librosa.cqt(data_downsampled, sr=sr, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE))
    np.save(path, data_cqt)

    return data_cqt.shape[1]


# Load csv files, transcribe the content to an array of 88 zeros and ones, and save it with the same name to the same location as a binary file
def process_csv(id, n_samples):
    #path_train = PATH_TRAIN_LABELS + str(id)
    #path_test = PATH_TEST_LABELS + str(id)
    #path = path_train if os.path.isfile(path_train + ".csv") else path_test
    path = PATH_LABELS + str(id)

    labels = np.zeros((n_samples, N_NOTES))
    labels_file = np.loadtxt(path + ".csv", delimiter=",", usecols=[0, 1, 3], skiprows=1)

    for line in labels_file:
        initial_frame = round(line[0] * SAMPLE_INDEXER)
        final_frame = round(line[1] * SAMPLE_INDEXER)
        pitch = round(line[2] - FMIN_MIDI_INDEX)
        labels[initial_frame:final_frame, pitch] = 1

    np.save(path, labels)


# Load metadata from csv file and filter out non-piano music files
metadata = pd.read_csv(PATH_METADATA)
metadata.drop(metadata[metadata["ensemble"] != "Solo Piano"].index, inplace = True)

ids = metadata["id"].to_numpy()

print("Files to process: ", len(ids))
for id in ids:
    print(f'Processing {id}...')
    n_samples = process_wav(id)
    process_csv(id, n_samples)
print("Finished")