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


import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)



class Dataset(data.Dataset):
    # Characterizes a dataset for PyTorch'
    def __init__(self, dataset_name, inputs, labels, transform=None, distractor=False, smoothing=False):
        # 'Initialization'
        self.labels = labels
        # self.list_IDs = list_IDs
        self.inputs = inputs
        self.smoothing = smoothing

        self.transform = transform
        self.distractor = distractor
        self.dataset_name = dataset_name
        # self.color_names = ['red','blue','green','yellow','violet','indigo','orange','purple','cyan','black']
        # self.color_class = []

        # for i in range(10):
        #     self.color_class.append(colors.to_rgb(self.color_names[i]))

    def __len__(self):
        # 'Denotes the total number of samples'
        return self.inputs.shape[0]



    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # ID = self.list_IDs[index]
        # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        img = self.inputs[index]

        if self.transform is not None:
            img = self.transform(img)
        
        y = int(self.labels[index])

        return img, y
    




import itertools as it


class GMULayer(nn.Module):
    def __init__(self,input_channels, output_channels, kernel_size, padding = 0, epsilon = 0.0001, num_slices=2,degree=4,exponent=True, normalize = True):
        super(GMULayer, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,kernel_size,kernel_size,num_slices))
        self.weight_bias = torch.nn.Parameter(torch.zeros(output_channels, input_channels,kernel_size,kernel_size))
        # self.adamantium_weights = torch.nn.Parameter(torch.zeros(output_channels, 2,num_slices))
        
        # torch.nn.init.xavier_normal_(self.adamantium_weights,gain=0.01)
        
        self.exponent = exponent
        self.kernel_size = kernel_size
        self.normalize = normalize
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.degree = degree 
        self.epsilon = epsilon
        self.padding = padding
        self.init_weights()


    def init_weights(self):
        
        # n = self.kernel_size*self.kernel_size*self.output_channels
        n = self.input_channels*self.output_channels
        stdv = 1. / np.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)
        # torch.nn.init.xavier_normal_(self.weights,gain=0.01)
        self.weight_bias.data.uniform_(0, 0.5)
        
        
    def forward(self, y2,train_status=True):

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
            X = X.view(X.shape[0],X.shape[1],X.shape[3]*X.shape[4],1)
            y = y - (X.repeat(y.shape[0],1,1,y.shape[3]))
            err = torch.mean((y)**2,dim=2)
            err = err.view(err.shape[0],err.shape[1],int(np.sqrt(err.shape[2])),int(np.sqrt(err.shape[2])))
            return (torch.exp(-err))
        
        if self.normalize:
            # GG = torch.std(y,dim=1)
            # y = y/GG.unsqueeze(1).repeat(1,y.shape[1],1)
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
        
        A = (torch.exp(-err)-np.exp(-1.0))/(np.exp(0)-np.exp(-1.0))
        return A-0.5
        

    
class SimpleGMULayer(nn.Module):
    def __init__(self,input_channels, output_channels, epsilon = 0.0001, num_slices=2,degree=4,exponent=True, normalize = True):
        super(SimpleGMULayer, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,num_slices))
        self.weight_bias = torch.nn.Parameter(torch.zeros(output_channels, input_channels))
        
        self.sigma = torch.nn.Parameter(torch.ones(1,output_channels,1,1))
        self.exponent = exponent
        self.normalize = normalize
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
        

    def forward(self, y,p):
            # print(self.weights.shape)
            y = y.squeeze()
            if len(y.shape)==1:
                y = y.unsqueeze(0)
           
            y = y + self.epsilon*torch.rand_like(y)
           
            y = y.unsqueeze(1).repeat(1,self.weight_bias.shape[0],1)
            y = y - (self.weight_bias.unsqueeze(0).repeat(y.shape[0],1,1))
            
            if self.num_slices == 0:
                err = torch.mean((y)**2,dim=2)
                err = err.unsqueeze(2).unsqueeze(3)
                return (torch.exp(-err))
            
            if self.normalize == True:
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
            err = err.unsqueeze(2).unsqueeze(3)
           
            return torch.exp(-err)
        
        
        
def regress_withgrad(x,y,p=1,normalize = True,exponent = False,**kwargs):
    
    #  x ~ samples x dimension
    #  y ~ samples x dimension (=1)
    if len(kwargs)==0:
        N = x.shape[0]
        if normalize == True:
            y = y/torch.std(y)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        X = torch.hstack([torch.ones((N, 1)), x])
        for i in range(p-1):
            X = torch.hstack([X, x**(i+2)])
        
        # print(X.T @ X)
        Mul_mat = torch.linalg.inv(X.T @ X) @ X.T
        W = Mul_mat @ y
        
        predicted = X @ W
        err = torch.mean((y-predicted)**2)
        
        if exponent:
            return torch.exp(-err),Mul_mat
        else:
            if normalize:
                return 1-err,Mul_mat
            else:
                return -err,Mul_mat
    else:
        N = x.shape[0]
        if normalize == True:
            y = y/torch.std(y)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        X = torch.hstack([torch.ones((N, 1)), x])
        for i in range(p-1):
            X = torch.hstack([X, x**(i+2)])
        
        # print(X.T @ X)
        W = kwargs['Mul_mat'] @ y
        
        predicted = X @ W
        err = torch.mean((y-predicted)**2)
        
        if exponent:
            return torch.exp(-err)
        else:
            if normalize:
                return 1-err
            else:
                return -err
        
        
    return 



from torcheval.metrics import R2Score


        
class ConvolutionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(ConvolutionLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (int(padding), int(padding))
        self.conv = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, stride=self.stride,
                              padding=self.padding)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x)




class NormalizedUnfoldConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, noise_std=1e-3):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.noise_std = noise_std

        # standard conv weights
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape

        # unfold into patches: shape (B, C*KH*KW, L)
        patches = F.unfold(x, kernel_size=self.kernel_size,
                           stride=self.stride, padding=self.padding)

        # add tiny Gaussian noise
        patches = patches + torch.randn_like(patches) * self.noise_std

        # normalize each patch (over channel*kernel dims)
        mean = patches.mean(dim=1, keepdim=True)
        std = patches.std(dim=1, keepdim=True)
        patches = (patches - mean) / std

        # fold back to image shape
        # shape: (B, C, H_out, W_out)
        H_out = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        x_norm = F.fold(patches, output_size=(H_out, W_out),
                        kernel_size=self.kernel_size,
                        stride=self.stride, padding=self.padding)

        # apply convolution
        return self.conv(x_norm)


import time


class Net_vanilla_CNN(nn.Module):
    def __init__(self,input_channels,layers,kernels,multiplier, epsilons = [0.0001,0.0001,0.0001],decay_regress=0,decay_errors = 0, classes = 10,use_bn = True,dropping=0,poly_order_init=5):
        super(Net_vanilla_CNN, self).__init__()
        
        self.m = multiplier
        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons 
        self.mode = 'normal'
        self.use_bn = True
        self.poly_order_init = poly_order_init
        self.padding = [2,0,0]
        # network layers
        self.convs = []
        self.bns = []
        self.Inds_weight = [] 
        self.Mul_mats = [0] 
        self.bns_rank = [] 
        self.decay_regress = decay_regress
        self.decay_errors = decay_errors 
        self.convs = [] 
        self.dropping = dropping
        
       
        
        if self.mode =='normal':
            self.conv1 = nn.Conv2d(1, self.m*32, 3, padding=1)
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
            self.fc1 = nn.Conv2d(self.m*64,self.m*64,1)
            self.fc2 = nn.Conv2d(self.m*64,10,1)
            
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
                self.conv3,
                self.bn3,
                nn.ReLU(inplace=True),
                self.mpool3,
                # #
                self.conv4,
                self.bn4,
                nn.ReLU(inplace=True),
                self.mpool4
                #
            )

        self.drop = nn.Dropout(p=self.dropping)
    
            
        self.relu = nn.ReLU(inplace=True)
        
    
        
    def forward(self, x):
       
        
        x = self.feat_net(x)                    
       
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)

        xm = xm.view(xm.size()[0], xm.size()[1])
        return xm,0
    
    
class Net_vanilla_CNN_convert_smaller(nn.Module):
    def __init__(self,input_channels,layers,kernels,multiplier,num_slices,
                 epsilons = [0.0001,0.0001,0.0001],decay_regress=0,decay_errors = 0, 
                 classes = 10,use_bn = True,dropping=0,poly_order_init=5):
        super(Net_vanilla_CNN_convert_smaller, self).__init__()
        
        self.m = multiplier 
        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons 
        self.mode = 'normal'
        self.use_bn = True
        self.poly_order_init = poly_order_init
        # self.padding = [2,0,0]
        # network layers
        self.convs = []
        self.bns = []
        self.Inds_weight = [] 
        self.Mul_mats = [0] 
        self.bns_rank = [] 
        self.decay_regress = decay_regress
        self.decay_errors = decay_errors 
        self.convs = [] 
        self.dropping = dropping
        self.gmu1 = GMULayer(input_channels, self.m*32, kernels[0],padding=int((kernels[0]-1)/2),epsilon = epsilons[0],num_slices=num_slices,degree=1,normalize=True)
        
       
        if self.mode =='normal':
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
            self.fc1 = nn.Conv2d(self.m*64,self.m*64,1)
            self.fc2 = nn.Conv2d(self.m*64,10,1)
            
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
                self.conv3,
                self.bn3,
                nn.ReLU(inplace=True),
                self.mpool3,
                # #
                self.conv4,
                self.bn4,
                nn.ReLU(inplace=True),
                self.mpool4
                #
            )
            
        self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
       
        if self.mode == 'normal':
            x = self.gmu1(x,net.training)
            x = self.feat_net(x)
            x_errs = x
                    
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)

        xm = xm.view(xm.size()[0], xm.size()[1])
        return xm,x_errs
    
    

class Net_vanilla_CNN_convert_evensmaller(nn.Module):
    def __init__(self,input_channels,layers,kernels,multiplier,num_slices,
                 epsilons = [0.0001,0.0001,0.0001],decay_regress=0,decay_errors = 0, 
                 classes = 10,use_bn = True,dropping=0,poly_order_init=5):
        super(Net_vanilla_CNN_convert_evensmaller, self).__init__()
        
        self.m = multiplier 
        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons 
        self.mode = 'normal'
        self.use_bn = True
        self.poly_order_init = poly_order_init
        # self.padding = [2,0,0]
        # network layers
        self.convs = []
        self.bns = []
        self.Inds_weight = [] 
        self.Mul_mats = [0] 
        self.bns_rank = [] 
        self.decay_regress = decay_regress
        self.decay_errors = decay_errors 
        self.convs = [] 
        self.dropping = dropping
        self.gmu1 = GMULayer(input_channels, self.m*32, kernels[0],padding=int((kernels[0]-1)/2),epsilon = epsilons[0],num_slices=num_slices,degree=1,normalize=True)
        
        # print('huhapa')
       
        if self.mode =='normal':
            self.conv1 = nn.Conv2d(1,self.m*32,3, padding=1)
            self.conv2 = nn.Conv2d(self.m*32, self.m*64, 3, padding=1)
            # self.conv3 = nn.Conv2d(self.m*64, self.m*64, 3, padding=1)
            # self.conv4 = nn.Conv2d(self.m*64, self.m*64, 3, padding=1)
            
            self.mpool1 = nn.MaxPool2d(2)
            self.mpool2 = nn.AdaptiveAvgPool2d((1, 1))
            # self.mpool3 = nn.MaxPool2d(2,padding=1)
            # self.mpool4 = nn.MaxPool2d(4)
            
            self.bn1 = nn.BatchNorm2d(self.m*32)
            self.bn2 = nn.BatchNorm2d(self.m*64)
            self.bn3 = nn.BatchNorm2d(self.m*64)
            self.bn4 = nn.BatchNorm2d(self.m*64)
            
            
            self.bnorm_fc = nn.BatchNorm2d(self.m*64)
            # self.fc1 = nn.Conv2d(self.m*64,self.m*64,1)
            self.fc2 = nn.Conv2d(self.m*64,10,1)
            
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
                # self.conv4,
                # self.bn4,
                # nn.ReLU(inplace=True),
                # self.mpool4
                # #
            )
            
        self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
       
        if self.mode == 'normal':
            x = self.gmu1(x,net.training)
            x = self.feat_net(x)
            x_errs = x
                    
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        # xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)

        xm = xm.view(xm.size()[0], xm.size()[1])
        return xm,x_errs
    
    
    
    
    
class Net_vanilla_CNN_convert(nn.Module):
    def __init__(self,input_channels,layers,kernels,num_slices,
                 epsilons = [0.0001,0.0001,0.0001],decay_regress=0,decay_errors = 0, 
                 classes = 10,use_bn = True,dropping=0,poly_order_init=5):
        super(Net_vanilla_CNN_convert, self).__init__()
        
        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons 
        self.mode = 'normal'
        self.use_bn = True
        self.poly_order_init = poly_order_init
        self.padding = [2,0,0]
        # network layers
        self.convs = []
        self.bns = []
        self.Inds_weight = [] 
        self.Mul_mats = [0] 
        self.bns_rank = [] 
        self.decay_regress = decay_regress
        self.decay_errors = decay_errors 
        self.convs = [] 
        self.dropping = dropping
        self.gmu1 = GMULayer(input_channels, 64, kernels[0],padding=int((kernels[0]-1)/2),epsilon = epsilons[0],num_slices=num_slices,degree=1,normalize=True)
        
       
        
        if self.mode =='normal':
            self.conv1 = nn.Conv2d(1,64,3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
            self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
            
            self.mpool1 = nn.AvgPool2d(2)
            self.mpool2 = nn.AvgPool2d(2)
            self.mpool3 = nn.MaxPool2d(2,padding=1)
            self.mpool4 = nn.MaxPool2d(4)
            
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(128)
            
            
            self.bnorm_fc = nn.BatchNorm2d(128)
            self.fc1 = nn.Conv2d(128,128,1)
            self.fc2 = nn.Conv2d(128,10,1)
            
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
                self.conv3,
                self.bn3,
                nn.ReLU(inplace=True),
                self.mpool3,
                # #
                self.conv4,
                self.bn4,
                nn.ReLU(inplace=True),
                self.mpool4
                #
            )

        self.drop = nn.Dropout(p=self.dropping)
    
            
        self.relu = nn.ReLU(inplace=True)
        
    
        
    def forward(self, x):
       
        if self.mode == 'normal':
            x = self.gmu1(x,net.training)
            x = self.feat_net(x)
            x_errs = x
                    
       
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        
        xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)

        xm = xm.view(xm.size()[0], xm.size()[1])
        return xm,x_errs
    

class Net_vanilla_NormalizedCNN_convert(nn.Module):
    def __init__(self,input_channels,layers,kernels,epsilons = [0.0001,0.0001,0.0001],decay_regress=0,decay_errors = 0, classes = 10,use_bn = True,dropping=0,poly_order_init=5):
        super(Net_vanilla_NormalizedCNN_convert, self).__init__()
        
        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons 
        self.mode = 'normal'
        self.use_bn = True
        self.poly_order_init = poly_order_init
        self.padding = [2,0,0]
        # network layers
        self.convs = []
        self.bns = []
        self.Inds_weight = [] 
        self.Mul_mats = [0] 
        self.bns_rank = [] 
        self.decay_regress = decay_regress
        self.decay_errors = decay_errors 
        self.convs = [] 
        self.dropping = dropping
        
       
        
        if self.mode =='normal':
            self.conv1 = NormalizedUnfoldConv2d(1, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
            self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
            
            self.mpool1 = nn.MaxPool2d(2)
            self.mpool2 = nn.MaxPool2d(2)
            self.mpool3 = nn.MaxPool2d(2,padding=1)
            self.mpool4 = nn.MaxPool2d(4)
            
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(128)
            
            
            self.bnorm_fc = nn.BatchNorm2d(128)
            self.fc1 = nn.Conv2d(128,128,1)
            self.fc2 = nn.Conv2d(128,10,1)
            
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
                self.conv3,
                self.bn3,
                nn.ReLU(inplace=True),
                self.mpool3,
                # #
                self.conv4,
                self.bn4,
                nn.ReLU(inplace=True),
                self.mpool4
                #
            )

        self.drop = nn.Dropout(p=self.dropping)
    
            
        self.relu = nn.ReLU(inplace=True)
        
    
        
    def forward(self, x):
       
        
        x = self.feat_net(x)                    
       
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)

        xm = xm.view(xm.size()[0], xm.size()[1])
        return xm,0
    
    


def train_network_normal(net,trainloader,testloader,test_labels,  init_rate,epochs,weight_decay):
    net = net
    net = net.cuda()
    net = net.train()
    optimizer = optim.Adam(net.parameters(), lr=init_rate, weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    init_epoch = 0
    all_train_losses = []
    bazinga = 0 
    train_loss_min = 9999
    for epoch in range(epochs):
        # print(epoch)
        # GG = time.time()
        train_loss = []
        loss_weights = [] 
       

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            allouts,x_errs = net(inputs)
            loss = criterion(allouts, labels.long()) #+ net.decay_errors*torch.mean(-x_errs)
            loss.backward()
            train_loss.append(loss.item())
            loss_weights.append(len(labels))
            
            optimizer.step()
       # 
        # print(time.time()-GG)
        
        all_train_losses.append(np.average(np.array(train_loss),weights=np.array(loss_weights)))
        if all_train_losses[-1] < train_loss_min:
            train_loss_min = copy(all_train_losses[-1])
            net_best = deepcopy(net)
       
    print(all_train_losses[-1])

        
    net_best = net_best.eval()
    return net_best,all_train_losses



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

def rand_another(label,label_max):
    array = torch.arange(label_max+1)
        
    array_removed = torch.cat([array[0:label], array[label+1:]]) 
    return array_removed[np.random.randint(0,len(array_removed))]



# def map_inf(data,atan_convert):

#     data = torch.atanh(atan_convert*data)
    
#     return data

def  load_data_and_generators(dataset_name,training_size,scale,rank_convert,atanh_convert,labelnoise):
    
    transform_train = transforms.Compose(
        [
          # torchvision.transforms.GaussianBlur(5, sigma=2.0),
          # torchvision.transforms.functional.rgb_to_grayscale
         transforms.ToTensor(),
         ])
    transform_test = transforms.Compose(
        [
            # torchvision.transforms.GaussianBlur(5, sigma=2.0),
         transforms.ToTensor(),
         ])

    
    
    if dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(root='./../../../data', train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.FashionMNIST(root='./../../../data', train=False,
                                                download=True, transform=transform_test)
        dataset = scale_dataset(dataset,scale)
        dataset_test = scale_dataset(dataset_test,scale)
        
        
        dataset.data = dataset.data.float().unsqueeze(1)
        dataset_test.data = dataset_test.data.float().unsqueeze(1)
    
    if dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root='./../../../data', train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.MNIST(root='./../../../data', train=False,
                                                download=True, transform=transform_test)
        dataset = scale_dataset(dataset,scale)
        dataset_test = scale_dataset(dataset_test,scale)
        
        
        dataset.data = dataset.data.float().unsqueeze(1)
        dataset_test.data = dataset_test.data.float().unsqueeze(1)

    elif dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)
        dataset = scale_dataset(dataset,scale,dataset_name)
        dataset_test = scale_dataset(dataset_test,scale,dataset_name)
        
        dataset.data = torch.permute(dataset.data,(0,3,1,2))
        dataset.targets = torch.from_numpy(np.array(dataset.targets))
        dataset.data = dataset.data.float()/255.0
        dataset_test.data = torch.permute(dataset_test.data,(0,3,1,2))
        dataset_test.targets = torch.from_numpy(np.array(dataset_test.targets))
        dataset_test.data = dataset_test.data.float()/255.0
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform_test)
        dataset.data = torch.permute(torch.from_numpy(dataset.data),(0,3,1,2))
        dataset.targets = torch.from_numpy(np.array(dataset.targets))
        dataset.data = dataset.data.float()/255.0
        dataset_test.data = torch.permute(torch.from_numpy(dataset_test.data),(0,3,1,2))
        dataset_test.targets = torch.from_numpy(np.array(dataset_test.targets))
        dataset_test.data = dataset_test.data.float()/255.0

    dataset.data = dataset.data[:training_size]
    dataset.targets = dataset.targets[:training_size]
   
    dataset.data = dataset.data.cuda()  #train_dataset.train_data is a Tensor(input data)
    dataset.targets = dataset.targets.cuda()
    
    # dataset_test.data = dataset_test.data.flatten(1,len(dataset_test.data.shape)-1).unsqueeze(2).unsqueeze(3)

    dataset_test.data = dataset_test.data.cuda()  #train_dataset.train_data is a Tensor(input data)
    dataset_test.targets = dataset_test.targets.cuda()
    
  
    
    my_dataset = Dataset(dataset_name, dataset.data, dataset.targets)
    my_dataset_test = Dataset(dataset_name, dataset_test.data, dataset_test.targets)

    trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                          shuffle=True,generator=torch.Generator(device='cuda'), num_workers=0)
    testloader = torch.utils.data.DataLoader(my_dataset_test, batch_size=batch_size,
                                          shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)
    
    

    
    return dataset,dataset_test, my_dataset,my_dataset_test,trainloader,testloader


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
    

# def denorm(batch, mean=[0.1307], std=[0.3081]):
#     """
#     Convert a batch of tensors to their original scale.

#     Args:
#         batch (torch.Tensor): Batch of normalized tensors.
#         mean (torch.Tensor or list): Mean used for normalization.
#         std (torch.Tensor or list): Standard deviation used for normalization.

#     Returns:
#         torch.Tensor: batch of tensors without normalization applied to them.
#     """
#     if isinstance(mean, list):
#         mean = torch.tensor(mean).to(device)
#     if isinstance(std, list):
#         std = torch.tensor(std).to(device)

#     return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def test_network_spike(net, testloader, test_labels,input_noise=0):
    net = net.eval()

    correct = torch.tensor(0)
    dataiter = iter(testloader)
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs + (input_noise*torch.randn_like(inputs)>0.1).float()*0.5
            all_outs,temp = net(inputs)
            predicted = torch.argmax(all_outs,1)

            correct = correct + torch.sum(predicted == labels)
    accuracy = float(correct) / float(len(test_labels))
    return accuracy

def test_network(net, testloader, test_labels,input_noise=0):
    net = net.eval()

    correct = torch.tensor(0)
    dataiter = iter(testloader)
    # total = 0 
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs + input_noise*torch.randn_like(inputs)
            all_outs,temp = net(inputs)
            predicted = torch.argmax(all_outs,1)
            correct = correct + torch.sum(predicted == labels)
    accuracy = float(correct) / float(len(test_labels))
    return accuracy

def test_network_corrupted(net, dataset_name, corruptions,atan_convert):
    net = net.eval()
    accuracy_list = [] 
    for corruption in corruptions:
        data_test = np.load('./'+ dataset_name+ '_c/'+ corruption +'/test_images.npy')
        labels_test = np.load('./'+ dataset_name+ '_c/'+ corruption +'/test_labels.npy')
        # data_test = (torch.from_numpy(data_test)).view(data_test.shape[0],
                                                     # int(data_test.size/data_test.shape[0]),1,1)
        labels_test = torch.from_numpy(labels_test)
        data_test = (torch.from_numpy(data_test)).float().squeeze().unsqueeze(1)/255.0
        # if atan_convert>0:
        #     data_test = map_inf(data_test, atan_convert)
        
        my_dataset_test = Dataset(dataset_name, data_test.cuda(), labels_test.cuda())

        testloader = torch.utils.data.DataLoader(my_dataset_test, batch_size=batch_size,
                                              shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)
        
        
        correct = torch.tensor(0)
        dataiter = iter(testloader)
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs
                inputs, labels = data
                all_outs,temp = net(inputs)
                predicted = torch.argmax(all_outs,1)

                correct = correct + torch.sum(predicted == labels)
        accuracy = float(correct) / float(len(labels_test))
        accuracy_list.append(accuracy)
        print("Corruption:",corruption, " Accuracy: ", accuracy)
    
    print("Mean accuracy:",np.mean(accuracy_list))
    return 0





import cv2


def test_network_dilate(net, testloader, test_labels,input_noise=0):
    net = net.eval()

    correct = torch.tensor(0)
    dataiter = iter(testloader)
    kernel = np.array([ [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1] ], dtype=np.float32)
    
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs + input_noise*torch.randn_like(inputs)
            inputs = inputs.view(inputs.shape[0],28,28)
            for j in range(inputs.shape[0]):
                inputs[j] = torch.from_numpy(cv2.dilate(inputs[j].detach().cpu().numpy(), kernel))
            inputs = inputs.view(inputs.shape[0],784,1,1)
            all_outs,temp = net(inputs)
            predicted = torch.argmax(all_outs,1)

            correct = correct + torch.sum(predicted == labels)
    accuracy = float(correct) / float(len(test_labels))
    return accuracy





def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
        
        
        
def test_network_with_adversary( model, testloader_adversary, epsilon ):

    correct = 0
    adv_examples = []
    
    # model = model.eval()
    # model = model.train()
    # model.apply(set_bn_eval)
    # model.use_bn = True
    model = model.eval()

    
    # model = model.eval
    # counter = 0 
    for datap, target in testloader_adversary:
        # get the inputs
        # optimizer = optim.Adam(data, lr=rate_array[0], weight_decay=weight_decay_array[0])

        
        datap = torch.tensor(datap, dtype=torch.float32, requires_grad=True) 
        output,temp = model(datap)
        # print(output.shape)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

       # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue
        # print(output)
        # print(target)
        loss = nn.CrossEntropyLoss()(output, target.long())
        # loss.requires_grad = True
        model.zero_grad()
        

        
        # Calculate gradients of model in backward pass
        loss.backward()
         
        # Collect ``datagrad``
        data_grad = datap.grad.data
        
        perturbed_data = fgsm_attack(datap, epsilon, data_grad)
        
        output,temp = model(perturbed_data)
        # print(final_pred)
    # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            # print('here')
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
    final_acc = correct/float(len(testloader_adversary))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(testloader_adversary)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
    

# def normalize_and_select_data(data,**kwargs):
    
#     if len (kwargs) ==0:
#         mean_vector = torch.mean(data,0)
#         std_vector =  torch.std(data,0)
#         means = mean_vector.repeat(data.shape[0],1,1,1)
#         std = std_vector.repeat(data.shape[0],1,1,1)
#         threshold = np.percentile(mean_vector.cpu(),40.0)
#         mask_vector = (mean_vector>threshold).float()
#         mask = mask_vector.repeat(data.shape[0],1,1,1)
#         data = (data - means)/std
#         data = data*mask 
#         return data, mean_vector, std_vector, mask_vector
    
#     else:
#         mean_vector = kwargs['mean_vector']
#         std_vector =  kwargs['std_vector']
#         mask_vector = kwargs['mask_vector']
#         mask = mask_vector.repeat(data.shape[0],1,1,1)
#         means = mean_vector.repeat(data.shape[0],1,1,1)
#         std = std_vector.repeat(data.shape[0],1,1,1)
#         data = (data -means)/std
#         data = data*mask
#         return data
from corruptions import *


def generate_corrupted_tensors(base_dataset, corruption_fn):
    corrupted_imgs = []
    corrupted_labels = []
    for i in range(len(base_dataset)):
        img, label = base_dataset[i]
        corrupted_img = torch.from_numpy(
            corruption_fn(255 * img[0].cpu().numpy())
        ).unsqueeze(0).float()
        corrupted_imgs.append(corrupted_img)
        corrupted_labels.append(label)
    return torch.stack(corrupted_imgs).cuda(), torch.tensor(corrupted_labels).cuda()

# Step 2: wrap into your Dataset class
class CorruptedDataset(torch.utils.data.Dataset):
    def __init__(self, name, data, targets):
        self.name = name
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    


def test_network_corrupted_loaders(net, corrupted_loaders):
    net.eval()
    accuracy_list = []
    with torch.no_grad():
        for name, loader in corrupted_loaders.items():
            correct = 0
            for inputs, labels in loader:
                # inputs, labels = inputs.cuda(), labels.cuda()
                all_outs, _ = net(inputs/255.0)
                predicted = torch.argmax(all_outs, 1)
                correct += (predicted == labels).sum().item()
            accuracy = correct / len(loader.dataset)
            accuracy_list.append(accuracy)
            print(f"Corruption: {name}, Accuracy: {accuracy:.4f}")
    print("Mean accuracy:", np.mean(accuracy_list))
    return np.mean(accuracy_list)


def test_network_corruptions(net, testloader, corruptions,atan_convert):
    net = net.eval()
    accuracy_list = [] 
    # corruptions = [motion_blur,shot_noise,translate,rotate,shear,scale,
    #                 spatter,zigzag]
    corruptions = [brightness,canny_edges,dotted_line,fog,impulse_noise,
                   motion_blur,shot_noise,spatter,zigzag]
    for corruption in corruptions:
        correct = torch.tensor(0)
        dataiter = iter(testloader)
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs
                inputs, labels = data
                for j in range(inputs.shape[0]):
                    inputs[j,0,:,:] = torch.from_numpy(corruption(255*inputs[j,0,:,:].cpu().numpy())).cuda()
                all_outs,temp = net(inputs)
                
                predicted = torch.argmax(all_outs,1)

                correct = correct + torch.sum(predicted == labels)
        accuracy = float(correct) / float(10000)
        accuracy_list.append(accuracy)
        # print("Corruption:",corruption, " Accuracy: ", accuracy)
    # 
    print("Mean accuracy:",np.mean(accuracy_list))
    return np.mean(accuracy_list)


def test_network_corruptions_with_bn_adaptation(net_orig, testloader, corruptions):
    
    accuracy_list = [] 
    corruptions = [brightness, canny_edges, dotted_line, fog, impulse_noise, motion_blur, shot_noise, spatter, zigzag]

    for corruption in corruptions:
        net = deepcopy(net_orig)
        net = net.train()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                layer.momentum = 0.95  # Update momentum for faster adaptation


        correct = torch.tensor(0)
        dataiter = iter(testloader)
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs
                inputs, labels = data
                
                # for temp_temp in range(10):
                  # Update BN statistics only
                
                # Switch to eval mode for inference
                
                
                for j in range(inputs.shape[0]):
                    inputs[j,0,:,:] = torch.from_numpy(corruption(255.0*inputs[j,0,:,:].cpu().numpy())).cuda()
                
                _ = net(inputs)
                net = net.eval()
                
                all_outs,duh = net(inputs)
                
                predicted = torch.argmax(all_outs,1)

                correct = correct + torch.sum(predicted == labels)
                net = net.train()
        accuracy = float(correct) / float(10000)
        accuracy_list.append(accuracy)
        print("Corruption:",corruption, " Accuracy: ", accuracy)
    
    print("Mean accuracy:",np.mean(accuracy_list))
    return 0



def test_network_corrupted(net, dataset_name, corruptions,atan_convert):
    net = net.eval()
    accuracy_list = [] 
    for corruption in corruptions:
        data_test = np.load('./'+ dataset_name+ '_c/'+ corruption +'/test_images.npy')
        labels_test = np.load('./'+ dataset_name+ '_c/'+ corruption +'/test_labels.npy')
        # data_test = (torch.from_numpy(data_test)).view(data_test.shape[0],
                                                     # int(data_test.size/data_test.shape[0]),1,1)
        labels_test = torch.from_numpy(labels_test)
        data_test = (torch.from_numpy(data_test)).float().squeeze().unsqueeze(1)/255.0
        # if atan_convert>0:
        #     data_test = map_inf(data_test, atan_convert)
        
        my_dataset_test = Dataset(dataset_name, data_test.cuda(), labels_test.cuda())

        testloader = torch.utils.data.DataLoader(my_dataset_test, batch_size=batch_size,
                                              shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)
        
        
        correct = torch.tensor(0)
        dataiter = iter(testloader)
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs
                inputs, labels = data
                all_outs,temp = net(inputs)
                predicted = torch.argmax(all_outs,1)

                correct = correct + torch.sum(predicted == labels)
        accuracy = float(correct) / float(len(labels_test))
        accuracy_list.append(accuracy)
        # print("Corruption:",corruption, " Accuracy: ", accuracy)
    
    print("Mean accuracy:",np.mean(accuracy_list))
    return np.mean(accuracy_list)




def stable_rank(W):

    W_np = W.detach().cpu().numpy()
    
    # Frobenius norm squared = sum of squares of all entries
    fro_sq = np.linalg.norm(W_np, 'fro')**2
    
    # Spectral norm = largest singular value
    spectral = np.linalg.norm(W_np, 2)
    
    if spectral == 0:
        return 0.0
    
    return float(fro_sq / (spectral**2))




def effective_rank_weights(weights):
    """
    weights: torch.Tensor of shape (N, 1, k, k, s)
    Returns: list of effective ranks for each of the N matrices
    """
    # Reshape to (N, k*k, s)
    N, _, k, _, s = weights.shape
    
    # Compute effective rank for each matrix
    ranks = [stable_rank(weights[i,0,:,:,:].reshape(k*k,s)) for i in range(N)]
    return ranks


import seaborn as sns

def plot_all_ranks(all_ranks, num_slices):
    # Professional style
    sns.set_style("whitegrid")
    colors = sns.color_palette("Set2", len(all_ranks))
    
    # Global font scaling
    plt.rcParams.update({
        "axes.titlesize": 34,
        "axes.labelsize": 30,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12
    })
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, ranks in enumerate(all_ranks):
        ax = axes[i]
        
        # Histogram with styling
        ax.hist(ranks, bins=10, color=colors[i], edgecolor='black', alpha=0.7, linewidth=0.7)
        ax.set_title(f"k = {num_slices[i]}", fontweight='bold')
        ax.set_xlabel("Stable Rank")
        ax.set_ylabel("Frequency")
        
        # Fixed x-axis range
        ax.set_xlim(0, 8)
        ax.grid(alpha=0.3)
        
        # Stats box
        mean_val = np.mean(ranks)
        std_val = np.std(ranks)
        min_val = np.min(ranks)
        max_val = np.max(ranks)
        
        textstr = '\n'.join((
            f'Mean: {mean_val:.2f}',
            f'Std: {std_val:.2f}',
            f'Min: {min_val:.2f}',
            f'Max: {max_val:.2f}',
        ))
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                fontsize=24, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # Hide unused subplot
    # fig.delaxes(axes[-1])
    
    # Add a global title
    # fig.suptitle("Stable Rank Distributions Across Slice Counts", fontsize=18, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()





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
    
    
        
    # corruptions = ['general']
    atanh_convert = 0 

    gc.collect()
    torch.cuda.empty_cache()
    
    dataset_name = "MNIST"
    batch_size = 200
    init_rate = 0.0005
    init_rate_crank = 0.01
    labelnoise = 0
    input_noise = 0 
    input_noise_array = [0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.25,0.3,0.35]
    epsilons = [ .001]
    gmu_epsilons = [0.001,0.00001,0]
    
    # test_transforms = ['scale','translate','rotate','']
    
    
    step_size = 10
    gamma_learning = 0.8
    total_epoch = 100
    total_epoch_crank = 100
    decay_normal = 0    
    decay_regress = 0
    decay_errors = 0
    dropping = 0 
    multiplier = int(1)
    rescale = 1.0
    decay_normal_crank = 0 
    
    layers = [25,50]
    kernels = [5,5]
    layers_crank = []
    
    training_size = 1000
    mode = 'regress_batch'
    use_bn = True
    rank_convert = False
    input_channels = 1
    global DEGREES
    DEGREE = [3,8]
    alpha=0.2
    
    ablation_slices = [1,2,3,4,5,6,7,8]
    average_corruption_accuracies = [] 
    normal_accuracies = [] 
    
    dataset,dataset_test,my_dataset,my_dataset_test,trainloader,testloader = load_data_and_generators(dataset_name,training_size,rescale,rank_convert,atanh_convert,labelnoise)

    
    # corruption_fns = [brightness, canny_edges, dotted_line, fog,
    #               impulse_noise, motion_blur, shot_noise,
    #               spatter, zigzag]
    
    # corrupted_loaders = {}
    # for corruption in corruption_fns:
    #     data, targets = generate_corrupted_tensors(dataset_test, corruption)
    #     corrupted_dataset = Dataset(corruption.__name__, data, targets)
    #     corrupted_loaders[corruption.__name__] = torch.utils.data.DataLoader(
    #         corrupted_dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         generator=torch.Generator(device='cuda'),
    #         num_workers=0
    #     )
        
    all_ranks = []
    
    for i in range(len(ablation_slices)):
        net = Net_vanilla_CNN_convert_smaller(input_channels,layers,kernels,multiplier,ablation_slices[i],gmu_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)

        net.load_state_dict(torch.load('./MNIST5k_gmuCNN_'+str(ablation_slices[i])+'slices.h5',weights_only=True))
        W = net.gmu1.weights 
        ranks = effective_rank_weights(W)
        
        all_ranks.append(ranks)
    
    


    plot_all_ranks(all_ranks,ablation_slices)
    
    
    # a = input("")
        
        
    
    
    # for i in range(len(ablation_slices)+1):
        
    #     acc_trials = [] 
        
    #     for j in range(5):
            
    #         if i == len(ablation_slices)+1:
    #             print("Version: Normal")
    #             net = Net_vanilla_CNN(input_channels,layers,kernels,multiplier, gmu_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)
    #         else:
    #             print("Version:", ablation_slices[i],"slices")
    #             net = Net_vanilla_CNN_convert_smaller(input_channels,layers,kernels,multiplier,ablation_slices[i],gmu_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)
            
            
    #         net.mode = 'normal'
            
    #         net,all_losses = train_network_normal(net,trainloader,testloader,my_dataset_test.labels, init_rate,total_epoch,decay_normal)
    #         # net.load_state_dict(torch.load('CNN_convert_3slice_1degree_FashionMNIST_[3, 8]degree_300epochs_[25, 50]layers60000data.h5'))
            
    #         # net.load_state_dict(torch.load('./MNIST_3kernel_NormalizedCNN_full.h5',weights_only=True))
        
        
    #         # net.load_state_dict(torch.load('./FashionMNIST10k_NormalCNN.h5',weights_only=True))
    
    #         acc =  test_network(net, testloader, my_dataset_test.labels, 0)
    #         acc_trials.append(acc)
            
    #         print('Test Accuracy:', acc)
            
    #         # accor = test_network_corrupted_loaders(net, corrupted_loaders)
        
    #     normal_accuracies.append(np.mean(acc_trials))
        # average_corruption_accuracies.append(accor)
        
        # accor = test_network_corruptions(net, testloader, corruptions,atanh_convert)
        # average_corruption_accuracies.append(accor)
    
        
        # test_network_corruptions_with_bn_adaptation(net, testloader, corruptions)    


    
    
