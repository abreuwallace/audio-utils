import numpy as np

def overlap_and_add(chunks, overlap=256, window_len=1024):
    W = window_len
    win_left_side = np.hanning(2 * overlap)[:overlap]
    win_right_side = np.hanning(2 * overlap)[overlap:]
    window = np.concatenate((win_left_side, np.ones(W - 2 * overlap), win_right_side))
    left_window = np.concatenate((np.ones(W - overlap), win_right_side))
    right_window = np.concatenate((win_left_side, np.ones(W - overlap)))    
    n_chunks = len(chunks)
    for i in range(n_chunks):
        #print(chunks.shape)
        if i == 0:
            y = (chunks[i] * left_window).reshape(-1,)
        else:
            x_chunk = chunks[i].reshape(-1,)
            if len(x_chunk) < W:
                x_chunk = np.pad(x_chunk, (0, W - len(x_chunk)), 'constant', constant_values=0)
                x_ola = x_chunk * right_window
            else:
                x_ola = x_chunk * window
            y = np.pad(y, (0, W - overlap), 'constant', constant_values=0)
            x_ola = np.pad(x_ola, (len(y) - len(x_ola), 0), 'constant', constant_values=0)
            y += x_ola
    return y
