import os
import scipy.io 
import numpy as np
import librosa
import librosa.display
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn import metrics
import torch
from torch import nn
from scipy.io.wavfile import read
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, out_channels = 4, stride = 1, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, out_channels = 10, stride = 1, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 20, stride = 1, kernel_size = 3)
        self.conv4 = nn.Conv2d(20, 40, stride = 1, kernel_size = 3)  
        self.fc1 = nn.Linear(172960, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x  = self.pool(F.relu(self.conv3(x)))
        x  = self.pool(F.relu(self.conv4(x)))       
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    fileNames = os.listdir('ZVUKOVI/')
    net = Net()