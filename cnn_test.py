import os
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
net.load_state_dict((torch.load('/content/drive/MyDrive/weightsCNN/run4weight20.pth')))
net=net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

X_train1, X_test, y_train1, y_test = train_test_split(l,target,test_size=0.15,random_state=0)
X_train, X_val, y_train, y_val =train_test_split(X_train1,y_train1,test_size=0.1765,random_state=0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002)
dataset = MyDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32,shuffle=True)
dataset1 = MyDataset(X_val, y_val)
val_loader = DataLoader(dataset1, batch_size=32,shuffle=True)
dataset2 = MyDataset(X_test, y_test)
test_loader = DataLoader(dataset2, batch_size=32,shuffle=True)

acc = 0.0
running_loss = 0.0
for i, data in enumerate(test_loader, 0):
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
print("test acc:",acc / len(dataset2))
print("test loss:",running_loss / len(dataset2))