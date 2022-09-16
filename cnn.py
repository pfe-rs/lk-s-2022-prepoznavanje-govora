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
from sklearn.model_selection import train_test_split

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

path1 = 'SPEKTROGRAMI/'
path2 = 'SPEKTROGRAMI CROPPED/'
l1 = os.listdir(path1)
target = [int(x[0]) for x in l1]
l = os.listdir(path2)
l = [path2 + fn for fn in l]

net = Net()
X_train1, X_test, y_train1, y_test = train_test_split(l,target,test_size=0.15,random_state=0)
X_train, X_val, y_train, y_val =train_test_split(X_train1,y_train1,test_size=0.1765,random_state=0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002)
dataset = MyDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32,shuffle=True)
dataset1 = MyDataset(X_val, y_val)
val_loader = DataLoader(dataset1, batch_size=32,shuffle=True)

val_accs = []
val_losses = []

for epoch in range(20):
    print(epoch+1)
    acc = 0.0
    running_loss = 0.0

    net.train()
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
        acc += int(torch.sum(labelice == labels))
    print("acc:",acc / len(dataset))
    print("loss:",running_loss / len(dataset))

    running_loss=0.0
    acc = 0.0

    net.eval()
    
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        labels2 = F.one_hot(labels % 10).type(torch.float)
        optimizer.zero_grad()
        outputs = net(inputs)
        labelice = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, torch.max(labels2, 1)[1])
        running_loss += loss.item()
        acc += int(torch.sum(labelice == labels))
    print("val_acc: ", acc / len(dataset1))
    print("val_loss: ", running_loss / len(dataset1))
    val_accs.append(acc / len(dataset1))
    val_losses.append(running_loss / len(dataset1))
    torch.save(net.state_dict(),f'/content/drive/MyDrive/weightsCNN/run4weight{epoch+1}.pth')