from numpy.lib.function_base import append
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import random
# import argparse
from tqdm import tqdm

from load_rotate import dataset
from load_rotate import test_data

# network model
class RegNagato(nn.Module):
    # def __init__(self):
    #     super(RegNagato, self).__init__()
    #     self.relu = nn.LeakyReLU(0.01)

    #     self.fc1 = nn.Linear(27, 336)
    #     self.fc2 = nn.Linear(336, 240)
    #     self.fc3 = nn.Linear(240, 168)
    #     self.fc4 = nn.Linear(168, 24)

    # def forward(self, x):
    #     x = self.fc1(x)
    #     x = self.relu(x)
    #     x = self.fc2(x)
    #     x = self.relu(x)
    #     x = self.fc3(x)
    #     x = self.relu(x)
    #     x = self.fc4(x)
        
    #     return x
    def __init__(self):
        super(RegNagato, self).__init__()
        self.relu = nn.LeakyReLU(0.01)

        self.fc1 = nn.Linear(27, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 24)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x
def eval(array):

    
    _x = torch.Tensor(array).float()
    # print("input --",_x)

    device = torch.device("cpu")
    net = RegNagato()
    net = net.to(device)
    net = net.eval() # 評価モードにする

    # パラメータの読み込み
    param = torch.load('rot_change3.pth')
    net.load_state_dict(param)
    output = net (_x)
    
    # print ("  Prediction   -- ", output.detach().numpy())

    return output.detach().numpy()


if __name__ == '__main__':

    eval() # evaluate mode
