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

class LSTM_SPLID(nn.Module):
    def __init__(self):
        super(LSTM_SPLID, self).__init__()
        self.lstm = nn.LSTM(16, 64, 2, batch_first=True) # Input size, hidden layers, out layers

        self.fc1 = nn.Linear(64 * 25, 128)
        self.fc2 = nn.Linear(64 * 25, 128)
        self.fc3 = nn.Linear(64 * 25, 128)

        self.fc_label1 = nn.Linear(128, 5) # output for node label 
        self.fc_label2 = nn.Linear(128, 4) # output for type label
        self.fc_label3 = nn.Linear(128, 8) # output for direction label

    def forward(self, x):

        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        x1 = F.relu(self.fc1(out[:, -1, :]))  # Output for label 1
        x2 = F.relu(self.fc2(out[:, -1, :]))  # Output for label 2
        x3 = F.relu(self.fc3(out[:, -1, :]))  # Output for label 3

        out_label1 = self.fc_label1(x1)
        out_label2 = self.fc_label2(x2)
        out_label3 = self.fc_label3(x3)

        return out_label1, out_label2, out_label3


# Create an instance of the CNN model
model = LSTM_SPLID()

# Print the model architecture
print(model)
