# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:34:32 2026

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:01:40 2026

@author: User
"""

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

from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.functional as F
import numpy as np
import sys, os
import torchvision.transforms.functional as TF
from torchvision.datasets import SVHN  # Import SVHN dataset
import matplotlib.pyplot as plt
import pickle
import random
from PIL import Image
import scipy
import time 
import gc
from copy import copy 
# import kornia
from scipy import stats
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
# +
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
# torch.set_float32_matmul_precision("high")


# Define Kornia augmentations

sys.path.append('./models') 


# +
import itertools as it


def train_network_normal(net, trainloader, testloader, init_rate, epochs):

    net = net.cuda()
    net = net.train()
    optimizer = optim.Adam(net.parameters(), lr=init_rate)

    criterion = nn.CrossEntropyLoss()
    
    all_train_losses = []

    for epoch in range(epochs):
        train_loss = []
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()
            
            allouts = net(inputs)
            loss = criterion(allouts, labels.long())
            loss.backward()
            
            optimizer.step()

            # Track loss item per batch
            train_loss.append(loss.item())

        # Compute average loss across all batches in the current epoch
        epoch_avg_loss = np.mean(train_loss)
        all_train_losses.append(epoch_avg_loss)

    print(f"Final Avg Train Loss: {epoch_avg_loss}")
        
    net = net.eval()
    return net, all_train_losses



def rand_another(label,label_max):
    array = torch.arange(label_max+1)
        
    array_removed = torch.cat([array[0:label], array[label+1:]]) 
    return array_removed[np.random.randint(0,len(array_removed))]

def scale_dataset(dataset_old,scale,dataset_name = 'MNIST'):
    if scale == 1.0:
        if dataset_name =='CIFAR10':
            dataset_old.data = torch.from_numpy(dataset_old.data)
        dataset_old.data = dataset_old.data.float()/255.0
        return dataset_old
    
    dataset = deepcopy(dataset_old)
    imresize = torchvision.transforms.Resize(int(dataset_old.data.shape[1]*scale))
    dataset.data = torch.zeros(dataset_old.data.shape[0],int(dataset_old.data.shape[1]*scale),int(dataset_old.data.shape[1]*scale))

    for i in range(dataset.data.shape[0]):
        J = Image.fromarray(np.uint8(dataset_old.data[i]))
        # I = transforms.ToTensor()(imresize(J.convert('L')))
        I = transforms.ToTensor()(imresize(J))
        # dataset.data[i] = I.permute(1,2,0)
        dataset.data[i] = I
        
        
    return dataset

class SimpleDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    
def load_data_and_generators(dataset_name, training_size, scale):
    
    transform_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(),  # Augmentation for better generalization
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])  # SVHN-specific normalization
    ])

# SVHN Test Transform (No Resize)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])  # SVHN-specific normalization
    ])

    if dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(root='./../../../data', train=True, download=True, transform=transform_train)
        dataset_test = torchvision.datasets.FashionMNIST(root='./../../../data', train=False, download=True, transform=transform_test)
        dataset = scale_dataset(dataset,1)
        dataset_test = scale_dataset(dataset_test,1)
        dataset.data = dataset.data.float().unsqueeze(1)
        dataset_test.data = dataset_test.data.float().unsqueeze(1)

    elif dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root='./../../../data', train=True, download=True, transform=transform_train)
        dataset_test = torchvision.datasets.MNIST(root='./../../../data', train=False, download=True, transform=transform_test)
        dataset = scale_dataset(dataset,1)
        dataset_test = scale_dataset(dataset_test,1)
        
        dataset.data = dataset.data.float().unsqueeze(1)
        dataset_test.data = dataset_test.data.float().unsqueeze(1)

    elif dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./../../../data', train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.CIFAR10(root='./../../../data', train=False,
                                                download=True, transform=transform_test)
        dataset = scale_dataset(dataset, scale, dataset_name)
        dataset_test = scale_dataset(dataset_test, scale, dataset_name)
        
        # Convert to Tensors and Permute
        train_x = torch.tensor(dataset.data).permute(0, 3, 1, 2).float()
        train_y = torch.tensor(dataset.targets)
        
        test_x = torch.tensor(dataset_test.data).permute(0, 3, 1, 2).float()
        test_y = torch.tensor(dataset_test.targets)

        # Wrap in our Custom Dataset instead of TensorDataset
        dataset = SimpleDataset(train_x, train_y)
        dataset_test = SimpleDataset(test_x, test_y)

    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        dataset_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        dataset.data = torch.permute(torch.tensor(dataset.data), (0, 3, 1, 2)).float() / 255.0
        dataset.targets = torch.tensor(dataset.targets)
        dataset_test.data = torch.permute(torch.tensor(dataset_test.data), (0, 3, 1, 2)).float() / 255.0
        dataset_test.targets = torch.tensor(dataset_test.targets)

    elif dataset_name == 'SVHN':
        dataset = SVHN(root='./data', split='train', download=True, transform=transform_train)
        dataset_test = SVHN(root='./data', split='test', download=True, transform=transform_test)
        dataset.data = torch.tensor(dataset.data).permute(0, 1, 2, 3).float() / 255.0
        dataset.targets = torch.tensor(dataset.labels)
        dataset_test.data = torch.tensor(dataset_test.data).permute(0, 1, 2, 3).float() / 255.0
        dataset_test.targets = torch.tensor(dataset_test.labels)
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported!")
        
    # dataset.data = dataset.data.float().reshape(dataset.data.size(0), -1)
    # dataset_test.data = dataset_test.data.float().reshape(dataset_test.data.size(0), -1)
    
    dataset.data = dataset.data[:training_size]
    dataset.targets = dataset.targets[:training_size]

    # Limit dataset size
    
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                              generator=torch.Generator(device='cuda'), num_workers=0)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                             generator=torch.Generator(device='cuda'), num_workers=0)

    return dataset, dataset_test, trainloader, testloader


def normalize_data(data,**kwargs):
    
    if len (kwargs) ==0:
        mean_vector = torch.mean(data,0)
        std_vector =  torch.std(data,0)
        means = mean_vector.repeat(data.shape[0],1,1,1)
        std = std_vector.repeat(data.shape[0],1,1,1)
        data = (data -means)/(std+0.00000001)
        return data, mean_vector,std_vector
    
    else:
        mean_vector = kwargs['mean_vector']
        std_vector =  kwargs['std_vector']
        means = mean_vector.repeat(data.shape[0],1,1,1)
        std = std_vector.repeat(data.shape[0],1,1,1)
        data = (data -means)/(std+0.00000001)
        return data



class GMUvision(nn.Module):
    def __init__(self,input_channels, output_channels, act_type='exp', epsilon = 0.0001, num_slices=2,degree=0):
        super(GMUvision, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,num_slices))
        self.weight_bias = torch.nn.Parameter(torch.zeros(output_channels, input_channels))
        
        # self.sigma = torch.nn.Parameter(torch.ones(1,output_channels,1,1))
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.degree = degree 
        self.epsilon = epsilon
        self.act_type = act_type
        
        self.init_weights()
        
    def init_weights(self):
        n = self.input_channels*self.output_channels
        stdv = 1. / np.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)
        self.weight_bias.data.uniform_(0, 0.5)
        # 

    def forward(self, y):
            # print(self.weights.shape)
            
            y = y.squeeze()
            if len(y.shape)==1:
                y = y.unsqueeze(0)
           
            y = y + self.epsilon*torch.rand_like(y)
           
            y = y.unsqueeze(1).repeat(1,self.output_channels,1)
            # y = y - (self.weight_bias.unsqueeze(0).repeat(y.shape[0],1,1))
            
            if self.num_slices == 0:
                y = y - (F.relu(self.weight_bias.unsqueeze(0).repeat(y.shape[0],1,1)))
                err = torch.mean((y)**2,dim=2)
                # err = err.unsqueeze(2).unsqueeze(3)
                if self.act_type == 'exp':
                    return torch.exp(-err)
                elif self.act_type == 'log':
                    return -torch.log(err)
                elif self.act_type == 'sqrt':
                    return torch.sqrt(1-err)
                elif self.act_type == 'lin':
                    return -err
                
            
            # y = y/(torch.std(y,2)).unsqueeze(2).repeat(1,1,self.weights.shape[1])
            
            X = self.weights
            for i in range(self.degree-1):
                X = torch.concat((X, self.weights**(i+2)),dim=2)
                
            # X = torch.concat((torch.ones((X.shape[0],X.shape[1],1)), X),dim=2)

            
            X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
            X_cov_inv = torch.linalg.inv(X_cov)
            M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
            
            
            W = torch.einsum('ijk,bik->ijb',M,y)
            
            pred_final = torch.einsum('bij,bjk->bik', X, W)
            
            pred_final = pred_final.permute(2,0,1)
            
            err = torch.mean((y-pred_final)**2,dim=2)
            # err = err.unsqueeze(2).unsqueeze(3)
           
            if self.act_type == 'exp':
                return torch.exp(-err)
            elif self.act_type == 'log':
                return -torch.log(err)
            elif self.act_type == 'sqrt':
                return torch.sqrt(1-err)
            elif self.act_type == 'lin':
                return -err


class GMUtabular(nn.Module):
    def __init__(self,input_channels, output_channels, epsilon = 0.0001, num_slices=2,degree=0):
        super(GMUtabular, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,num_slices))
        self.weight_bias = torch.nn.Parameter(torch.zeros(output_channels, input_channels))
        
        self.sigma = torch.nn.Parameter(torch.ones(1,output_channels,1,1))
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.degree = degree 
        self.epsilon = epsilon
        
        self.init_weights()
        
    def init_weights(self):
        n = self.input_channels*self.output_channels
        stdv = 1. / np.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)
        self.weight_bias.data.uniform_(0, 0.5)
        

    def forward(self, y):
            # print(self.weights.shape)
            y = y.squeeze()
            if len(y.shape)==1:
                y = y.unsqueeze(0)
           
            y = y + self.epsilon*torch.rand_like(y)
           
            y = y.unsqueeze(1).repeat(1,self.weight_bias.shape[0],1)
            y = y - (self.weight_bias.unsqueeze(0).repeat(y.shape[0],1,1))
            
            if self.num_slices == 0:
                err = torch.mean((y)**2,dim=2)
                # err = err.unsqueeze(2).unsqueeze(3)
                return -err
                
            
            y = y/(torch.std(y,2)).unsqueeze(2).repeat(1,1,self.weights.shape[1])
            
            X = self.weights
            for i in range(self.degree-1):
                X = torch.concat((X, self.weights**(i+2)),dim=2)
            

            
            X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
            X_cov_inv = torch.linalg.inv(X_cov)
            M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
            
            
            W = torch.einsum('ijk,bik->ijb',M,y)
            
            pred_final = torch.einsum('bij,bjk->bik', X, W)
            
            pred_final = pred_final.permute(2,0,1)
            
            err = torch.mean((y-pred_final)**2,dim=2)
            # err = err.unsqueeze(2).unsqueeze(3)
            
            return -err

# +


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

        



def test_network(net, testloader):
    net = net.eval()

    correct = torch.tensor(0)
    dataiter = iter(testloader)
    test_labels = testloader.dataset.targets
    # total = 0 
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            all_outs = net(inputs)
            predicted = torch.argmax(all_outs,1)
            correct = correct + torch.sum(predicted == labels)
    accuracy = float(correct) / float(len(test_labels))
    return accuracy




class Matched_MLP(nn.Module):
    def __init__(self, input_dim, target_params, num_classes=10, normalize=False, eps=1e-4):
        super(Matched_MLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.normalize = normalize
        self.eps = eps
        
        # 1. Calculate required hidden dimension H
        # Formula: P = H*(I + O + 3) + O  ==>  H = (P - O) / (I + O + 3)
        denominator = input_dim + num_classes + 3  # +3 accounts for layer1 bias + bn1 weight & bias
        
        self.hidden_dim = max(1, int((target_params - num_classes) // denominator))
        
        # 2. Instantiate layers
        self.layer1 = nn.Linear(input_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        if self.normalize:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True, unbiased=False)
            x = (x - mean) / (std + self.eps)
            
        x = self.layer1(x)
        x = self.layer2(F.relu(self.bn1(x)))
        
        return x

# --- Helper Class for the Normalized Perceptron ---
class NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5):
        super(NormalizedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.eps = eps

    def forward(self, x):
        # Apply intra-sample normalization (mean=0, std=1 across dimensions)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)
        return self.linear(x)




def count_parameters(model):
    """Helper method to verify total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        
        # Hidden GMU layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        # self.drop = nn.Dropout(0.2) if use_dropout else nn.Identity()
        
        # Final fully connected classifier
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Hidden GMU + BN
        out = self.fc1(x)
        out = F.relu(self.bn1(out))
        
        # Dropout
        # out = self.drop(out)
        
        # Final linear output (logits)
        out = self.fc_out(out)
        
        return out
    
class ActGMUMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_slices, degree, act ='exp'):
        super(ActGMUMLP, self).__init__()
        
        # Hidden GMU layer
        self.gmu1 = GMUvision(input_dim, hidden_dim, act, epsilon=epsilon, num_slices=num_slices, degree=degree)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        # self.drop = nn.Dropout(0.2) if use_dropout else nn.Identity()
        
        # Final fully connected classifier
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Hidden GMU + BN
        out = self.gmu1(x)
        out = F.relu(self.bn1(out))
        
        # Dropout
        # out = self.drop(out)
        
        # Final linear output (logits)
        out = self.fc_out(out)
        
        return out



class Net_vanilla_CNN(nn.Module):
    def __init__(self,input_channels,multiplier, classes = 10):
        super(Net_vanilla_CNN, self).__init__()
        
        self.m = multiplier
    
    
        
        self.conv1 = nn.Conv2d(input_channels, self.m*32, 3, padding=1)
        self.conv2 = nn.Conv2d(self.m*32, self.m*64, 3, padding=1)
        self.conv3 = nn.Conv2d(self.m*64, self.m*64, 3, padding=1)
        self.conv4 = nn.Conv2d(self.m*64, self.m*64, 3, padding=1)
        
        self.mpool1 = nn.MaxPool2d(2)
        self.mpool2 = nn.MaxPool2d(2)
        self.mpool3 = nn.MaxPool2d(2,padding=1)
        self.mpool4 = nn.MaxPool2d(4)
        
        self.bn1 = nn.BatchNorm2d(self.m*32)
        self.bn2 = nn.BatchNorm2d(self.m*64)
        self.bn3 = nn.BatchNorm2d(self.m*64)
        self.bn4 = nn.BatchNorm2d(self.m*64)
        
        
        self.bnorm_fc = nn.BatchNorm2d(self.m*64)
        self.fc1 = nn.Conv2d(self.m*64,classes,1)
        # self.fc2 = nn.Conv2d(self.m*64,classes,1)
        
        self.feat_net = nn.Sequential(

            self.conv1,
            self.mpool1,
            self.bn1,
            nn.ReLU(inplace=True),
            

            self.conv2,
            self.bn2,
            nn.ReLU(inplace=True),
            self.mpool2,
            #
            # self.conv3,
            # self.bn3,
            # nn.ReLU(inplace=True),
            # self.mpool3,
            # # #
            self.conv4,
            self.bn4,
            nn.ReLU(inplace=True),
            self.mpool4
            #
        )

            
        self.relu = nn.ReLU(inplace=True)
        
    
        
    def forward(self, x):
       
        
        x = self.feat_net(x)                    
       
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        # xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc1(xm)

        xm = xm.view(xm.size()[0], xm.size()[1])
        return xm
    
    
    
class Net_vanilla_CNN_convert_smaller(nn.Module):
    def __init__(self,input_channels,multiplier,num_slices,
                 epsilon =0.0001,classes = 10, act = 'exp'):
        super(Net_vanilla_CNN_convert_smaller, self).__init__()
        
        self.m = multiplier 
        
        self.gmu1 = GMULayer(input_channels, self.m*32, 3,padding=1,epsilon = epsilon,
                             num_slices=num_slices,degree=1,act = act)
        
   
        self.conv1 = nn.Conv2d(1,self.m*32,3, padding=1)
        self.conv2 = nn.Conv2d(self.m*32, self.m*64, 3, padding=1)
        self.conv3 = nn.Conv2d(self.m*64, self.m*64, 3, padding=1)
        self.conv4 = nn.Conv2d(self.m*64, self.m*64, 3, padding=1)
        
        self.mpool1 = nn.MaxPool2d(2)
        self.mpool2 = nn.MaxPool2d(2)
        self.mpool3 = nn.MaxPool2d(2,padding=1)
        self.mpool4 = nn.MaxPool2d(4)
        
        self.bn1 = nn.BatchNorm2d(self.m*32)
        self.bn2 = nn.BatchNorm2d(self.m*64)
        self.bn3 = nn.BatchNorm2d(self.m*64)
        self.bn4 = nn.BatchNorm2d(self.m*64)
        
        
        self.bnorm_fc = nn.BatchNorm2d(self.m*64)
        # self.fc1 = nn.Conv2d(self.m*64,self.m*64,1)
        self.fc1 = nn.Conv2d(self.m*64,10,1)
        
        self.feat_net = nn.Sequential(
            # self.conv1,
            self.mpool1,
            self.bn1,
            # nn.ReLU(inplace=True),
            
            self.conv2,
            self.bn2,
            nn.ReLU(inplace=True),
            self.mpool2,
            #
            # self.conv3,
            # self.bn3,
            # nn.ReLU(inplace=True),
            # self.mpool3,
            # # #
            self.conv4,
            self.bn4,
            nn.ReLU(inplace=True),
            self.mpool4
            #
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        x = self.gmu1(x)
        x = self.feat_net(x)
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        # xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc1(xm)

        xm = xm.view(xm.size()[0], xm.size()[1])
        return xm
    

class GMULayer(nn.Module):
    def __init__(self,input_channels, output_channels, kernel_size, padding = 0, epsilon = 0.0001, num_slices=2,degree=4, act = 'exp'):
        super(GMULayer, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,kernel_size,kernel_size,num_slices))
        self.weight_bias = torch.nn.Parameter(torch.zeros(output_channels, input_channels,kernel_size,kernel_size))
        # self.adamantium_weights = torch.nn.Parameter(torch.zeros(output_channels, 2,num_slices))
        self.act_type = act
        # torch.nn.init.xavier_normal_(self.adamantium_weights,gain=0.01)
        
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.degree = degree 
        self.epsilon = epsilon
        self.padding = padding
        self.init_weights()


    def init_weights(self):
        
        n = self.input_channels*self.output_channels
        stdv = 1. / np.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)
        # torch.nn.init.xavier_normal_(self.weights,gain=0.01)
        self.weight_bias.data.uniform_(0, 0.5)
        
        
    def forward(self, y2):

        y = nn.Unfold((self.weights.shape[2],self.weights.shape[3]),padding=self.padding)(y2)        

        y = y + self.epsilon*torch.randn_like(y)
        # y = torch.hstack([y,torch.ones(y.shape[0],1,y.shape[2]),torch.zeros(y.shape[0],1,y.shape[2])])
        

        if self.num_slices == 0:
            y= y - y.mean(dim=1).unsqueeze(1).repeat(1,y.shape[1],1)
            GG = torch.std(y,dim=1)  
            y = y/GG.unsqueeze(1).repeat(1,y.shape[1],1)
            y = y.unsqueeze(1).repeat(1,self.output_channels,1,1)
            X = self.weight_bias
            X = X.unsqueeze(0)
            X = X.view(X.shape[0],X.shape[1],X.shape[2]*X.shape[3]*X.shape[4],1)
            y = y - (X.repeat(y.shape[0],1,1,y.shape[3]))
            err = torch.mean((y)**2,dim=2)
            err = err.view(err.shape[0],err.shape[1],int(np.sqrt(err.shape[2])),int(np.sqrt(err.shape[2])))
            # print(err.min(), err.max())
            if self.act_type == 'exp':
                return torch.exp(-err)
            elif self.act_type == 'log':
                return -torch.log(err+0.0001)
            elif self.act_type == 'sqrt':
                return torch.sqrt(1 - torch.minimum(torch.tensor(0.99), err))
            elif self.act_type == 'lin':
                return -err
        
        # if self.normalize:
        #     # GG = torch.std(y,dim=1)
        #     # y = y/GG.unsqueeze(1).repeat(1,y.shape[1],1)
        # y= y - y.mean(dim=1).unsqueeze(1).repeat(1,y.shape[1],1)
        std = y.std(dim = 1, keepdim = True)
        y = y/std
            
            
        
        X = self.weights 
        X = X.view(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3],self.num_slices)
        
        
        # for i in range(self.degree-1):
        #     X = torch.concat((X, X[:,:,0:self.num_slices]**(i+2)),dim=2)
        
        
        
        X = torch.concat((torch.ones((X.shape[0],X.shape[1],1),requires_grad=False), X),dim=2)
        
        # X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
        X_cov = torch.matmul(X.transpose(1, 2), X)
        
        X_cov_inv = torch.linalg.inv(X_cov)
        # M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
        M = torch.matmul(X_cov_inv, X.transpose(1, 2))   # [B, s, d]

        
        W = torch.einsum('ijk,akb->aijb',M,y)
       
        pred_final = torch.einsum('bec,abcd->abed', X, W)   
        
        
        err = torch.mean((y.unsqueeze(1).repeat(1,pred_final.shape[1],1,1)-pred_final)**2,dim=2)
        
        err = err.view(err.shape[0],err.shape[1],int(np.sqrt(err.shape[2])),int(np.sqrt(err.shape[2])))
        # err = err/err.detach().max()
        
        if self.act_type == 'exp':
            return torch.exp(-err)
        elif self.act_type == 'log':
            return -torch.log(err+0.0001)
        elif self.act_type == 'sqrt':
            return torch.sqrt(1-err)
        elif self.act_type == 'lin':
            return -err
        
        # A = (torch.exp(-err)-np.exp(-1.0))/(np.exp(0)-np.exp(-1.0))
        # return A-0.5
    
if __name__ == "__main__":
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    
#     corruptions = ['general']

    gc.collect()
    torch.cuda.empty_cache()
    
    dataset_name = "MNIST"
    
    init_rate = 0.001
    
    rescale = 1.0
    
    training_size = 10000
    output_channels = 10
    hidden_dim = 200
    output_dim = 10
    batch_size = 200
    total_epoch = 200
    epsilon = 0.0001
    multiplier = 1
    
    # ablation_slices = [1,2,3,4,5,6,7,8]
    # average_corruption_accuracies = [] 
    normal_accuracies = [] 
    # act_list = ['exp','log', 'lin']
    act_list = ['sqrt']
  
    # net = VGG('VGG16-GMU',num_slices=3)
    dataset,dataset_test,trainloader,testloader = load_data_and_generators(dataset_name,training_size,rescale)
    
    input_channels = dataset.data.shape[1]
    
    test_accuracies = {
        'GMUvision': {},
        'GMUtabular': {},
        'Perceptron_Standard': None,
        'Perceptron_Normalized': None
    }
    
    print("Starting Experiment Loop...\n" + "="*40)

    # ---------------------------------------------------------
    # Baseline 1: Standard Perceptron
    # ---------------------------------------------------------
    # print("Training Standard 3 Layer CNN(Baseline)...")

    # net = Net_vanilla_CNN(input_channels,multiplier)
    # net, _ = train_network_normal(net, trainloader, testloader, init_rate, total_epoch)

    # acc_perceptron = test_network(net, testloader)
    # test_accuracies['Perceptron_Standard'] = acc_perceptron
    # print(f"Test Accuracy 3 Layer CNN: {acc_perceptron:.4f}")
    
    # ---------------------------------------------------------
    # Main Loop (Varying Degree from 1 to 5)
    # ---------------------------------------------------------
    for act in act_list:
        for s in [0,3]:  # s now represents the 'degree'
            print(f"\n--- Training configurations for slices = {s} and act = {act} ---")
            
            # ---------------------------------------------------------
            # 1. GMUvision
            # ---------------------------------------------------------
            print(f"Training GMUvision (slices={s})...")
            net_vision = Net_vanilla_CNN_convert_smaller(input_channels,multiplier,num_slices=s,
                         epsilon = epsilon,classes = output_channels, act = act)

            net_vision, _ = train_network_normal(net_vision, trainloader, testloader, init_rate, total_epoch)
            
            acc_vision = test_network(net_vision, testloader)
            test_accuracies['GMUvision'][s] = acc_vision
            print(f"Test Accuracy GMUvision: {acc_vision:.4f}")



  


