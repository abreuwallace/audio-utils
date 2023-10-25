#for type hint
from __future__ import annotations
import glob 
import torch 
import torchaudio

def get_filenames_in_folder(path: list[str], format='wav') -> list[str]:
    if format == 'wav':
        filenames = glob.glob(path + '*.wav')
    elif format == 'mp3':
        filenames = glob.glob(path + '*.mp3')

    return filenames

def load_multiple_audio(names: list[str]) -> list[torch.FloatTensor]:
    audio = []
    for name in names:
        x, _ = torchaudio.load(name)
        if x.shape[0] > 1:
            audio.append(torch.mean(x, dim=0).view(1, 1, -1))
        else:
            audio.append(x.view(1, 1, -1))
    return audio