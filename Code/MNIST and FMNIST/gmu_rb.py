# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 12:52:53 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:22:37 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:29:29 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 23:41:00 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:21:35 2024

@author: User
"""
import torch
from torch.utils import data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.functional as F
import numpy as np
import sys, os
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import pickle
import random
from PIL import Image
import scipy
import time 
import gc
from copy import copy 
import kornia
from scipy import stats

from torch.utils.data import Dataset, DataLoader

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


import torch
from torch.utils.data import Dataset, DataLoader

class PrecomputedDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        x = self.inputs[index]
        if self.transform is not None:
            x = self.transform(x)
        y = self.labels[index]
        return x, y





import itertools as it


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 use_dropout=True):
        super(SimpleMLP, self).__init__()
        
        # Hidden fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        # self.drop = nn.Dropout(0.2) if use_dropout else nn.Identity()
        
        # Final fully connected classifier
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Hidden layer
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Dropout
        # out = self.drop(out)
        
        # Final linear output (logits)
        out = self.fc_out(out)
        
        return out


    
class SimpleGMUfc():
    def __init__(self,input_channels, output_channels, threshold, num_slices=2):
        super(SimpleGMUfc, self).__init__()
        
        self.weights = torch.zeros(output_channels, input_channels,num_slices)
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.threshold = threshold
        
        for s in range(num_slices):
            # assign a unique input channel index for this slice
            self.weights[:, s, s] = 1.0
        

    def output(self, y):
            # print(self.weights.shape)
            y = y.squeeze()
            if len(y.shape)==1:
                y = y.unsqueeze(0)
                      
            y = y.unsqueeze(1).repeat(1,self.weights.shape[0],1)
            
            X = self.weights

            
            X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
            X_cov_inv = torch.linalg.inv(X_cov)
            M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
            
            
            W = torch.einsum('ijk,bik->ijb',M,y)
            
            pred_final = torch.einsum('bij,bjk->bik', X, W)
            
            pred_final = pred_final.permute(2,0,1)
            
            dist = torch.sqrt(torch.mean((y-pred_final)**2,dim=2))
            # err = err.unsqueeze(2).unsqueeze(3)
            return dist>self.threshold
            
        
        
class uRBF(nn.Module):
    def __init__(self,input_channels, output_channels, num_slices):
        super(uRBF, self).__init__()
        
        self.weight_bias = torch.nn.Parameter(torch.zeros(output_channels, input_channels))
        
        self.sigma = torch.nn.Parameter(torch.ones(1,output_channels,1,1))
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        
        self.init_weights()
        
    def init_weights(self):
        self.weight_bias.data.uniform_(0, 0.5)
        self.sigma.data.uniform_(0.5, 1.5)


    def forward(self, y):
        y = y.squeeze()
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
    
        # expand input to match weight_bias
        y = y.unsqueeze(1).repeat(1, self.weight_bias.shape[0], 1)
        y = y - self.weight_bias.unsqueeze(0).repeat(y.shape[0], 1, 1)
    
        # squared distance
        err = torch.mean(y**2, dim=2)
    
        # divide by 2 * sigma^2
        out = torch.exp(-err / (2 * self.sigma.squeeze()**2))
    
        return out

        
        
class RBFMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                  num_slices, normalize=True, use_dropout=True):
        super(RBFMLP, self).__init__()
        
        # Hidden GMU layer
        self.rbf_in = uRBF(input_dim, hidden_dim,num_slices)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(1)
        # Dropout
        # self.drop = nn.Dropout(0.2) if use_dropout else nn.Identity()
        
        # Final fully connected classifier
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        # Hidden GMU + BN
        out = self.rbf_in(x)
        out = self.bn1(out)
        
        # Dropout
        # out = self.drop(out)
        
        # Final linear output (logits)
        out = self.fc_out(out)
        
        return out




def train_network_normal(net, trainloader, init_rate, epochs, weight_decay, device="cuda"):
    net = net.to(device)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=init_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")  # sum so we can weight manually

    all_train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        total_samples = 0

        for inputs, labels in trainloader:
            labels =labels.float().view(-1, 1) 

            optimizer.zero_grad()
            logits = net(inputs)

            # sum reduction gives total loss over batch
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # accumulate weighted loss
            batch_size = labels.size(0)
            epoch_loss += loss.item()
            total_samples += batch_size

        # weighted average loss over all samples
        avg_loss = epoch_loss / total_samples
        all_train_losses.append(avg_loss)

    print("Final weighted train loss:", all_train_losses[-1])
    return net, all_train_losses



def test_network(net, testloader):
    net.eval()
    TP = TN = FP = FN = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            labels = labels.float().view(-1, 1)

            logits = net(inputs)                  # (batch, 1)
            probs = torch.sigmoid(logits)         # convert to probabilities
            preds = (probs > 0.5).long()          # threshold at 0.5

            # confusion matrix components
            TP += ((preds == 1) & (labels == 1)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()

    # avoid division by zero
    recall_pos = TP / (TP + FN) if (TP + FN) > 0 else 0
    recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0

    balanced_accuracy = 0.5 * (recall_pos + recall_neg)
    return balanced_accuracy




import seaborn as sns


import math

def hypersphere_volume(m, R):
    return (math.pi ** (m / 2)) / math.gamma(m / 2 + 1) * (R ** m)

def error_estimate(d, k, R, N):

    V_dk = hypersphere_volume(d - k, R)
    V_d  = hypersphere_volume(d, R)
    N = N - (np.power(N,(k-1)/d)*k)
    return V_dk - N * V_d


def balanced_error_estimate(d, k, R, N):

    V_dk = hypersphere_volume(d - k, R)
    V_d  = hypersphere_volume(d, R)
    print(V_dk/V_d)
    N = N - (np.power(N,(k-1)/d)*k)
    return 0.5- (N * V_d/(2*V_dk))


    
    
    # a = input("")
        
        
def gen_labels(inputs, input_loader,label_gen):
    
    all_labels = []
    for batch in input_loader:
        x = batch[0]                     # shape (200, n_dim)
        y = label_gen.output(x).long()   # generate labels for this batch
        all_labels.append(y)
    
    labels = torch.cat(all_labels, dim=0) 
    
    return labels


def gen_loaders(n_samples, n_dim,batch_size,num_slices, threshold):
    inputs_train = torch.rand(n_samples, n_dim) - 0.5
    inputs_test = torch.rand(n_samples, n_dim) - 0.5
    
    input_dataset_train = TensorDataset(inputs_train)
    input_loader_train = DataLoader(input_dataset_train, batch_size=200, shuffle=False)
    
    input_dataset_test = TensorDataset(inputs_test)
    input_loader_test = DataLoader(input_dataset_test, batch_size=200, shuffle=False)
    
    
    
    label_gen = SimpleGMUfc(input_channels=n_dim, output_channels=1, threshold=threshold,num_slices=num_slices)
    
    train_labels = gen_labels(inputs_train, input_loader_train, label_gen)
    test_labels = gen_labels(inputs_test, input_loader_test, label_gen)
    
    print(torch.mean(train_labels.float()))
    
    dataset_train = PrecomputedDataset(inputs_train, train_labels)
    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,generator=torch.Generator(device='cuda'), num_workers=0 )
    
    dataset_test = PrecomputedDataset(inputs_test, test_labels)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True,generator=torch.Generator(device='cuda'), num_workers=0 )

    return trainloader, testloader


if __name__ == "__main__":
    
    
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    # corruptions = ['brightness','canny_edges','dotted_line','fog','glass_blur','identity',
    #                 'impulse_noise','motion_blur','shot_noise','translate','rotate','shear','scale',
    #                 'spatter','zigzag']
    # corruptions = ['brightness','canny_edges','dotted_line','fog','impulse_noise',
    #                'motion_blur','shot_noise','spatter','zigzag']
    
    
    n_samples = 10000
    n_dim = 10
    threshold = 0.25
    num_slices = [1,2,3,4,5]
    hidden_dim_range = np.arange(10,200,10)
    N_list = [1,10,100,200]
    output_dim = 1
    # Initialize your label generator
    
    


    #Training params
    init_rate = 0.0005
    total_epoch = 200
    batch_size = 400
    decay_normal = 0 
    
    
    final_list = []
    # Test one batch
    for N in N_list:
        acc_list = [] 
        for i in range(len(num_slices)):
            
            trainloader, testloader = gen_loaders(n_samples, n_dim,batch_size,num_slices[i], threshold)
            net_rbf  = RBFMLP(n_dim, N, output_dim,num_slices[i])
            net_rbf,all_losses = train_network_normal(net_rbf,trainloader,
                                                      init_rate,total_epoch,decay_normal)
                    
            acc =  test_network(net_rbf,testloader)
            print(balanced_error_estimate(n_dim, num_slices[i], threshold,N ))
            print('Test Error GMU-MLP:', 1-acc)
            acc_list.append(acc)
        final_list.append(acc_list)
        
        
            
    # corruptions = ['general']

    
     
    # net = SimpleMLP(input_channels, hidden_dim, output_dim)
    # net,all_losses = train_network_normal(net,trainloader,testloader,my_dataset_test.labels, init_rate,total_epoch,decay_normal)
    
    # acc =  test_network(net, testloader, my_dataset_test.labels, 0)
    # print('Test Accuracy MLP:', acc)
    
        
        # accor = test_network_corruptions(net, testloader, corruptions,atanh_convert)
        # average_corruption_accuracies.append(accor)
    
        
        # test_network_corruptions_with_bn_adaptation(net, testloader, corruptions)    


    
    
