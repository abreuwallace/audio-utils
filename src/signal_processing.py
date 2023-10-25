#for type hint
from __future__ import annotations
import numpy as np

def to_mono(x: np.ndarray) -> np.ndarray:
    if len(x.shape) > 1:
        return np.mean(x, axis=-1)
    else:
        return x
    
def overlap_and_add(chunks: list[np.ndarray], overlap=256, window_len=1024) -> np.ndarray:
    W = window_len
    win_left_side = np.bartlett(2 * overlap)[:overlap]
    win_right_side = np.bartlett(2 * overlap)[overlap:]
    window = np.concatenate((win_left_side, np.ones(W - 2 * overlap), win_right_side))
    left_window = np.concatenate((np.ones(W - overlap), win_right_side))
    right_window = np.concatenate((win_left_side, np.ones(W - overlap)))    
    n_chunks = len(chunks)
    for i in range(n_chunks):
        if i == 0:
            y = (chunks[i].reshape(-1,) * left_window)
        else:
            x_chunk = chunks[i].reshape(-1,)
            if len(x_chunk) < W or i == n_chunks - 1:
                end_pad = W - len(x_chunk)
                x_chunk = np.pad(x_chunk, (0, end_pad), 'constant', constant_values=0)
                x_ola = x_chunk * right_window
            else:
                x_ola = x_chunk * window
            y = np.pad(y, (0, W - overlap), 'constant', constant_values=0)
            x_ola = np.pad(x_ola, (len(y) - len(x_ola), 0), 'constant', constant_values=0)
            y += x_ola
    return y
