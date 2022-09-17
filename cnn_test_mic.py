import os
import numpy as np
from scipy.io import wavfile
from PIL import Image
import os.path
import torchsummary
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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, out_channels = 4, stride = 1, kernel_size=3)
        self.c1_bn = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(4, out_channels = 6, stride = 1, kernel_size=3)
        self.c2_bn = nn.BatchNorm2d(6)
        
        self.conv3 = nn.Conv2d(6, 10, stride = 1, kernel_size = 3)
        self.c3_bn = nn.BatchNorm2d(10)
        
        self.conv4 = nn.Conv2d(10, 20, stride = 1, kernel_size = 3)  
        self.c4_bn = nn.BatchNorm2d(20)
        
        self.fc1 = nn.Linear(86480, 128)
        self.fc1_bn = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.fc2_bn = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.pool(self.c1_bn(F.relu(self.conv1(x))))
        x = self.pool(self.c2_bn(F.relu(self.conv2(x))))
        x  = self.pool(self.c3_bn(F.relu(self.conv3(x))))
        x  = self.pool(self.c4_bn(F.relu(self.conv4(x))))       
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.softmax(x)
        return x

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = torch.tensor(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        t = transforms.ToTensor()
        filename = self.data[index]
        x = Image.open('SPEKTROGRAMI CROP TEST/'+filename)
        x = t(x)
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.data)

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.savefig('SPEKTROGRAMI CROP TEST/testspectrogram.png')
sd.default.device[0] = 2
fs = 44100 
length = 3
print("PRICAJ")
recording = sd.rec(frames=fs * length, samplerate=fs, blocking=True, channels=1)
sd.wait()
i=1
wavfile.write(f'ZVUKOVI_TEST/snimak{i}.wav', fs, recording)
print("KRAJ")

scale, sr = librosa.load('ZVUKOVI_TEST/snimak1.wav')
HOP_SIZE = 512
FRAME_SIZE = 2048

S_scale = librosa.stft(scale,n_fft=FRAME_SIZE,hop_length=HOP_SIZE)
Y_scale = np.abs(S_scale) ** 2
Y_log_scale = librosa.power_to_db(Y_scale)
plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")
im = Image.open('SPEKTROGRAMI CROP TEST/testspectrogram.png')
im1 = im.crop((315,122,1861,888))
im1.show()
im1.save('SPEKTROGRAMI CROP TEST/testspectrogram.png')
l = os.listdir('SPEKTROGRAMI CROP TEST/')
net = Net()
net.load_state_dict((torch.load('run4weight20.pth', map_location=torch.device('cpu'))))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
net.eval()

y_test = [3]
dataset = MyDataset(l,y_test)
loader = DataLoader(dataset,batch_size=32)
inputs, labels = next(iter(loader))
optimizer.zero_grad()
outputs = net(inputs)
labelice = torch.argmax(outputs, dim=1)
optimizer.step()
labelice = np.array(labelice)
print(labelice[0])