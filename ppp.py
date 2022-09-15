import os
import matplotlib.pyplot as plt
from PIL import Image
import os.path
import librosa
import librosa.display
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from projekat_cnn import Net
import torch.optim as optim


def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = torch.tensor(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        t = transforms.ToTensor()
        filename = self.data[index]
        x = Image.open(filename)
        x = t(x)
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.data)

path = 'ZVUKOVI/'
path1 = 'SPEKTROGRAMI/'
path2 = 'SPEKTROGRAMI CROPPED/'
# fileNames = os.listdir(path1)
# for i in fileNames:
#     img = Image.open(path1+i)
#     box = (315,122,1861,888)
#     img2 = img.crop(box)
#     img2.save('a'+str(i))
l1 = os.listdir(path1)
target = [int(x[0]) for x in l1]
l = os.listdir(path2)
l = [path2 + fn for fn in l]


dataset = MyDataset(l, target)
loader = DataLoader(dataset, batch_size=32,shuffle=True)
# for batch_ndx, sample in enumerate(loader):
#     print(sample[0].shape)

fileNames = os.listdir('ZVUKOVI/')


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

for epoch in range(5):
    sumica = 0
    acc = 0
    running_loss = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        labels2 = F.one_hot(labels % 10).type(torch.float)
        optimizer.zero_grad()
        outputs = net(inputs)
        labelice = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, torch.max(labels2, 1)[1])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        sumica = sumica + int(torch.sum(labelice == labels))
        if(i%10==0):
            if(i!=0):
                running_loss/=10
            print("loss: ",running_loss)
            print("acc: ", sumica/((i + 1) * len(labels)))
            running_loss = 0