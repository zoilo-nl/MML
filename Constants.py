from librosa import note_to_hz, note_to_midi

PATH_METADATA = "musicnet_metadata.csv"

PATH_DATA = "musicnet/data/"
PATH_LABELS = "musicnet/labels/"

PATH_TRAIN_DATA = "musicnet/train_data/"
PATH_TRAIN_LABELS = "musicnet/train_labels/"
PATH_TEST_DATA = "musicnet/test_data/"
PATH_TEST_LABELS = "musicnet/test_labels/"

ORIGINAL_SR = 44100 # original sample rate in Hz
TARGET_SR = 16000 # target sample rate in Hz
FMIN = note_to_hz("A0") # lowest frequency of the model
FMIN_MIDI_INDEX = note_to_midi("A0") # MIDI index of the note corresponding to lowest frequency
N_NOTES = 88
BINS_PER_NOTE = 3
BINS_PER_OCTAVE = 12 * BINS_PER_NOTE
N_BINS = N_NOTES * BINS_PER_NOTE
HOP_LENGTH = 512

FRAME_LENGTH = HOP_LENGTH / TARGET_SR # Frame duration in seconds
FRAME_RATE = 1 / FRAME_LENGTH # Frames per second

FRAC_SR = TARGET_SR / ORIGINAL_SR
SAMPLE_INDEXER = FRAC_SR / HOP_LENGTH