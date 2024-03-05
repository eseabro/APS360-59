import numpy as np
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import shutil 

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_SPLID(nn.Module):
    def __init__(self, input_channels):
        super(CNN_SPLID, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = nn.Linear(64 * 25, 128)
        self.fc2 = nn.Linear(64 * 25, 128)
        self.fc3 = nn.Linear(64 * 25, 128)

        self.fc_label1 = nn.Linear(128, 5) # output for node label 
        self.fc_label2 = nn.Linear(128, 4) # output for type label
        self.fc_label3 = nn.Linear(128, 8) # output for direction label

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28) # Flattening for the fully connected layer
        x = F.relu(self.fc(x))
        
        x1 = F.elu(self.fc1(x))
        x2 = F.elu(self.fc2(x))
        x3 = F.elu(self.fc3(x))
        
        # Output for each label with appropriate activation functions
        out_label1 = F.softmax(self.fc_label1(x1), dim=1)
        out_label2 = torch.sigmoid(self.fc_label2(x2))
        out_label3 = F.relu(self.fc_label3(x3))
        
        return out_label1, out_label2, out_label3


# Create an instance of the CNN model
model = CNN_SPLID()

# Print the model architecture
print(model)
