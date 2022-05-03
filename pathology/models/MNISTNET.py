"""
Created on May 2, 2022.
mnistNet.py

@author: oliver

https://github.com/tayebiarasteh/
"""


import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F


class mnistNet(nn.Module):
    def __init__(self):
        super(mnistNet, self).__init__()
        self.dense = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(512, 256)
        self.dense2 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 2)

    def forward(self, x):
        # x = torch.flatten(x, 1)
        x = self.dense(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.dense1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.dense2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.dense3(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.dense4(x)
        output = x

        return output
