import os
import numpy as np
from scipy.io import wavfile
from PIL import Image
import os.path
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from projekat_cnn import Net
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sounddevice as sd
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
sd.default.device[0] = 2
fs = 44100 
length = 2
print("PRICAJ")
recording = sd.rec(frames=fs * length, samplerate=fs, blocking=True, channels=1)
sd.wait()
i=1
wavfile.write(f'ZVUKOVI_TEST/snimak{i}.wav', fs, recording)
print("KRAJ")

scale, sr = librosa.load('ZVUKOVI_TEST/snimak1.wav')
FRAME_SIZE = 2048
HOP_SIZE = 512
S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
Y_scale = np.abs(S_scale) ** 2
Y_log_scale = librosa.power_to_db(Y_scale)
plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")
plt.savefig('SPEKTROGRAMI CROP TEST/testspectrogram.png')
im = Image.open('SPEKTROGRAMI CROP TEST/testspectrogram.png')
im1 = im.crop((315,122,1861,888))