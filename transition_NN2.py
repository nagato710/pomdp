from numpy.lib.function_base import append
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torch.utils.data import Dataset
import torchbnn as bnn
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import random
# import argparse
from tqdm import tqdm

from load import dataset
from load import test_data

class RegNagato(nn.Module):
    def __init__(self):
        super(RegNagato, self).__init__()
        self.relu = nn.LeakyReLU(0.01)

        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=21, out_features=256)
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=256, out_features=256)
        self.fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=256, out_features=18)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
# network model
def eval(array,model_num):

    # print("type_check")
    # print(array)

    input =  torch.from_numpy(np.array(array)).float()
    # output =  torch.from_numpy(np.array(testset.t)).float()
    
    device = torch.device("cpu")
    net = RegNagato()
    net = net.to(device)
    net = net.eval() # 評価モードにする

    net.load_state_dict(torch.load("model" + str(model_num) + ".pth",map_location=torch.device('cpu')))
    net.eval()


    for i in range(10):
        y_predict = net(input)
    
    # print ("  Prediction   -- ", output.detach().numpy())

    return y_predict.detach().numpy()


if __name__ == '__main__':

    eval() # evaluate mode
