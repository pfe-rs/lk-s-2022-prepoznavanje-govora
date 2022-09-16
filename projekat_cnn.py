import os
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

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
if __name__ == "__main__":
    fileNames = os.listdir('ZVUKOVI/')
    net = Net()