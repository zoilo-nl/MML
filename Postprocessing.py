def post_process(y, min_rest_len, min_note_len=-1):
    for i in range(y.shape[1]):
        rest_count = 0
        note_count = 0
        for j in range(y.shape[0]):
            # If the preceeding and the current event is a rest, keep counting rests
            if y[j, i] == 0 and (counting_rests == True or j + 1 < min_rest_len):
                counting_rests = True
                rest_count = rest_count + 1
            # If the preceeding and the current event is a note, keep counting notes
            elif y[j, i] == 1 and (counting_rests == False or j + 1 < min_note_len):
                counting_rests = False
                note_count = note_count + 1
            elif y[j, i] == 0:
                counting_rests = True
                # If it is a first rest, and the amount of notes preceeding it is too little
                if note_count < min_note_len:
                    y[j - note_count : j, i] = 0
                    rest_count = rest_count + note_count + 1
                    note_count = 0
                # If it is a first rest, and the amount of notes preceeding it is long enough
                else:
                    rest_count = 1
            elif y[j, i] == 1:
                counting_rests = False
                # If it is a note note, and the amount of rests preceeding it is too little
                if rest_count < min_rest_len:
                    y[j - rest_count : j, i] = 1
                    note_count = note_count + rest_count + 1
                    rest_count = 0
                # If it is a first note, and the amount of rests preceeding it is long enough
                else:
                    note_count = 1
    return y
