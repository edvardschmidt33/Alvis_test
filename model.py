import torch, torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)   # [batch, 1, 28, 28] -> [batch, 16, 24, 24]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # -> [batch, 32, 20, 20]
        self.pool = nn.MaxPool2d(2, 2)                 # -> [batch, 32, 10, 10]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)  # -> [batch, 64, 8, 8]
        
        self.fc1 = nn.Linear(64 * 8 * 8, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 8 * 8)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.size = 16
#         size = self.size

#         self.W1 = nn.Parameter(0.1* torch.rand(size,1, 5, 5 ))
#         self.b1 = nn.Parameter(0.1* torch.ones(size))

#         self.W2 = nn.Parameter(0.1* torch.rand(size // 2 , size, 7, 7 ))
#         self.b2 = nn.Parameter(0.1* torch.ones(size // 2))

#         self.W3 = nn.Parameter(0.1 * torch.rand(size * 2, size // 2, 3, 3 ))
#         self.b3 = nn.Parameter(0.1 * torch.ones(size * 2))

#         # Linear layers
#         self.W4 = nn.Parameter(0.1 * torch.rand((size *2 ) * 16 * 16, 200))
#         self.b4 = nn.Parameter(0.1 * torch.ones(200))

#         self.W5 = nn.Parameter(0.1* torch.rand(200, 10))
#         self.b5 = nn.Parameter(0.1* torch.ones(10))

#     def forward(self, X):
#         Q1 = F.relu(F.conv2d(X, self.W1, bias=self.b1, stride=1, padding =0))
#         Q2 = F.relu(F.conv2d(Q1, self.W2, bias = self.b2, stride = 1, padding = 0))
#         Q3 = F.relu(F.conv2d(Q2, self.W3, bias = self.b3, stride = 1, padding = 0))
#         #flatten x
#         Q3flat = Q3.view(-1, self.size *2 *16*16)
#         Q5 = F.relu(Q3flat.mm(self.W4) + self.b4)

#         Z = Q5.mm(self.W5) + self.b5
#         return Z


