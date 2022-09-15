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
        self.conv1 = nn.Conv2d(4, out_channels = 8, stride = 1, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, out_channels = 16, stride = 1, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, stride = 1, kernel_size = 3)
        self.conv4 = nn.Conv2d(32, 64, stride = 1, kernel_size = 3)  
        self.fc1 = nn.Linear(276736, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x  = self.pool(F.relu(self.conv3(x)))
        x  = self.pool(F.relu(self.conv4(x)))       
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    fileNames = os.listdir('ZVUKOVI/')
    net = Net()