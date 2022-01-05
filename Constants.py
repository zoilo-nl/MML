from librosa import note_to_hz, note_to_midi

PATH_METADATA = "musicnet_metadata.csv"

PATH_WAV = "musicnet/data/"
PATH_CSV = "musicnet/labels/"

PATH_DATA = "musicnet/data_1B/"
PATH_LABELS = "musicnet/labels_1B/"

PATH_TRAIN_DATA = "musicnet/train_data/"
PATH_TRAIN_LABELS = "musicnet/train_labels/"
PATH_TEST_DATA = "musicnet/test_data/"
PATH_TEST_LABELS = "musicnet/test_labels/"

ORIGINAL_SR = 44100 # original sample rate in Hz
TARGET_SR = 16000 # target sample rate in Hz
FMIN = note_to_hz("A0") # lowest frequency of the model
FMIN_MIDI_INDEX = note_to_midi("A0") # MIDI index of the note corresponding to lowest frequency
N_NOTES = 88 # number of notes of the piano
BINS_PER_NOTE = 1 # number of bins the CQT will compute for every note
BINS_PER_OCTAVE = 12 * BINS_PER_NOTE # number of bins of the chromatic scale
N_BINS = N_NOTES * BINS_PER_NOTE # number of frequency bins computed by the CQT
HOP_LENGTH = 512 # number of samples between successive CQT columns

FRAME_LENGTH = HOP_LENGTH / TARGET_SR # frame duration in seconds
FRAME_RATE = 1 / FRAME_LENGTH # frames per second

FRAC_SR = TARGET_SR / ORIGINAL_SR
SAMPLE_INDEXER = FRAC_SR / HOP_LENGTH