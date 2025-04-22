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

