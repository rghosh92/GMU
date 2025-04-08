# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 16:20:40 2025

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:30:09 2025

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

import kornia.augmentation as K


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
from torchvision.transforms.functional import to_tensor,to_pil_image

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



class Hypernetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Hypernetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        return self.network(x)
 
class HyperLinearLayer(nn.Module):
    def __init__(self, hyper_input_dim, target_input_dim, target_output_dim):
        super(HyperLinearLayer, self).__init__()
        # Hypernetwork to generate both weights and biases
        self.hypernetwork_weights = Hypernetwork(hyper_input_dim, 10, target_input_dim * target_output_dim)
        self.hypernetwork_biases = Hypernetwork(hyper_input_dim, 10, target_output_dim)  # Generate biases separately
        self.target_input_dim = target_input_dim
        self.target_output_dim = target_output_dim

    def forward(self, x, hyper_input):
        # Generate weights dynamically using the hypernetwork
        global_hyper_input = hyper_input.mean(dim=0)  # Mean across batch (or customize this step)

        weights = self.hypernetwork_weights(global_hyper_input)
        weights = weights.view(self.target_output_dim, self.target_input_dim)  # Reshape weights for linear layer

        # Generate biases dynamically using the hypernetwork
        biases = self.hypernetwork_biases(global_hyper_input)  # Output biases directly

        # Perform the linear layer operation
        output = nn.functional.linear(x, weights, biases)  # Linear transformation with weights and biases
        return output

    
    
class HyperGMULayer(nn.Module):
    def __init__(self,input_channels, output_channels, kernel_size, padding = 0, epsilon = 0.0001, num_slices=2, hyper_hidden_dim=10):
        super(HyperGMULayer, self).__init__()
        
        self.hypernet = Hypernetwork(input_channels*784, hyper_hidden_dim, 
                                     output_channels * input_channels * kernel_size * kernel_size * num_slices)
        # self.adamantium_weights = torch.nn.Parameter(torch.zeros(output_channels, 2,num_slices))
        # torch.nn.init.xavier_normal_(self.adamantium_weights,gain=0.01)
        
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.epsilon = epsilon
        self.padding = padding
        self.iter =  it.combinations(np.arange(num_slices), 2)

    def forward(self, y2,train_status=True):
        
        global_hyper_input = y2.mean(dim=0)  # Mean across batch (or customize this step)

        # Generate convolution weights dynamically using the hypernetwork
        weights = self.hypernet(global_hyper_input)  # One set of weights for the entire batch
        weights = weights.view(self.output_channels, self.input_channels, 
                       self.kernel_size, self.kernel_size, self.num_slices) 

        y = nn.Unfold((weights.shape[2],weights.shape[3]),padding=self.padding)(y2)        
        y = y + self.epsilon*torch.randn_like(y)
        # y = torch.hstack([y,torch.ones(y.shape[0],1,y.shape[2]),torch.zeros(y.shape[0],1,y.shape[2])])
        GG = torch.std(y,dim=1)  
        y = y/GG.unsqueeze(1).repeat(1,y.shape[1],1)
        
        X = weights 
        X = X.view(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3],self.num_slices)
        
        
        
        
        X = torch.concat((torch.ones((X.shape[0],X.shape[1],1),requires_grad=False), X),dim=2)
        
        X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
        X_cov_inv = torch.linalg.inv(X_cov+ decay_regress)
        M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
        
        W = torch.einsum('ijk,akb->aijb',M,y)
       
        pred_final = torch.einsum('bec,abcd->abed', X, W)   
        
        
        err = torch.mean((y.unsqueeze(1).repeat(1,pred_final.shape[1],1,1)-pred_final)**2,dim=2)
        
        err = err.view(err.shape[0],err.shape[1],int(np.sqrt(err.shape[2])),int(np.sqrt(err.shape[2])))
        # err = err/err.detach().max()
        
        A = (torch.exp(-err)-np.exp(-1.0))/(np.exp(0)-np.exp(-1.0))
        return A-0.5

class GMULayer(nn.Module):
    def __init__(self,input_channels, output_channels, kernel_size, padding = 0, epsilon = 0.0001, num_slices=2,degree=4,exponent=True, normalize = True):
        super(GMULayer, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,kernel_size,kernel_size,num_slices))
        # self.adamantium_weights = torch.nn.Parameter(torch.zeros(output_channels, 2,num_slices))
        torch.nn.init.xavier_normal_(self.weights,gain=0.1)
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
        self.iter =  it.combinations(np.arange(num_slices), 2)

    def forward(self, y2,train_status=True):

        y = nn.Unfold((self.weights.shape[2],self.weights.shape[3]),padding=self.padding)(y2)        
        y = y + self.epsilon*torch.randn_like(y)
        # y = torch.hstack([y,torch.ones(y.shape[0],1,y.shape[2]),torch.zeros(y.shape[0],1,y.shape[2])])
        
        GG = torch.std(y,dim=1)  
        y = y/GG.unsqueeze(1).repeat(1,y.shape[1],1)
        
        X = self.weights 
        X = X.view(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3],self.num_slices)
        
        
        for i in range(self.degree-1):
            X = torch.concat((X, X[:,:,0:self.num_slices]**(i+2)),dim=2)
        
        
        
        X = torch.concat((torch.ones((X.shape[0],X.shape[1],1),requires_grad=False), X),dim=2)
        
        X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
        X_cov_inv = torch.linalg.inv(X_cov+ decay_regress)
        M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
        
        W = torch.einsum('ijk,akb->aijb',M,y)
       
        pred_final = torch.einsum('bec,abcd->abed', X, W)   
        
        
        err = torch.mean((y.unsqueeze(1).repeat(1,pred_final.shape[1],1,1)-pred_final)**2,dim=2)
        
        err = err.view(err.shape[0],err.shape[1],int(np.sqrt(err.shape[2])),int(np.sqrt(err.shape[2])))
        # err = err/err.detach().max()
        
        A = (torch.exp(-err)-np.exp(-1.0))/(np.exp(0)-np.exp(-1.0))
        return A-0.5
        
    
    
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


        
    
# def rankCorrelation_variants_1D(X,Y,variant='chatterjee_normal',**kwargs):
    
#     # X is 2D and Y assumed is 1D torch tensor 
#     if variant == "chatterjee_normal":
#         rank_Y = get_rank_argsort(Y).float()
        
#         CC_array = torch.zeros(X.shape[0])
        
#             n = len(X[i])
#             ind_X = torch.argsort(X[i])
        
#             Xsorted_Y= rank_Y[ind_X]
#             XsY0 = Xsorted_Y[:-1]
#             XsY1 = Xsorted_Y[1:]
#             diff_sum = torch.sum(torch.abs(XsY0-XsY1))
#             CC = 1 - (3*diff_sum/((n**2)-1))
#             CC_array[i] = CC
        
#         return CC_array 
#     if variant == "chatterjee_norank":        
#         min_Y = float(kwargs['min_y'])
#         max_Y = float(kwargs['max_y'])
#         rank_Y = (Y - min_Y)/max_Y
#         CC_array = torch.zeros(X.shape[0])

#         for i in range(X.shape[0]):
#             n = len(X[i])
#             ind_X = torch.argsort(X[i])
#             Xsorted_Y= rank_Y[ind_X]
#             XsY0 = Xsorted_Y[:-1]
#             XsY1 = Xsorted_Y[1:]
#             diff_sum = torch.sum(torch.abs(XsY0-XsY1))
#             CC = 1 - (3*n*diff_sum/((n**2)-1))
#             CC_array[i] = CC
#         return CC_array


class BiasLayer(torch.nn.Module):
    def __init__(self,shape) -> None:
        super().__init__()
        bias_value = torch.randn(shape)*0.1
        self.bias_layer = torch.nn.Parameter(bias_value)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(x + self.bias_layer)
    
    

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


class Net_vanilla_cnn_small(nn.Module):
    def __init__(self,input_channels,layers,kernel_sizes,decay_regress=0, classes = 10,use_bn = True,dropping=0):
        super(Net_vanilla_cnn_small, self).__init__()

        # kernel_sizes = [5, 3, 3]
        pads = np.array([0,0,0])
        pads = pads.astype(int)
        hidden_ratio = 1
        layers = [int(30*hidden_ratio), int(60*hidden_ratio),int(60*hidden_ratio)]
        self.post_filter = False
        # network layers
        self.conv1 = ConvolutionLayer(input_channels, layers[0], [kernel_sizes[0], kernel_sizes[0]], stride=1, padding=pads[0])
        self.conv2 = ConvolutionLayer(layers[0], layers[1], [kernel_sizes[1], kernel_sizes[1]], stride=1,
                                      padding=pads[1])
        # self.conv5 = ConvolutionLayer(layers[3], layers[4], [kern1el_sizes[2], kernel_sizes[2]], stride=1, padding=pads[2])

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1 = nn.BatchNorm2d(layers[0])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn2 = nn.BatchNorm2d(layers[1])
        # self.pool5 = nn.MaxPool2d(2)
        # self.bn5 = nn.BatchNorm2d(layers[4])
        self.fc1 = nn.Conv2d(layers[1], layers[2], 1)
        self.fc1bn = nn.BatchNorm2d(1500)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout2d(0.7)
        # self.fc2 = nn.Conv2d(layers[2], 10, 1)
        self.fc_direct = nn.Conv2d(1500, classes, 1)
        
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x_checkpoint = self.bn2(x)
        # print(x.shape)
        # print(x.shape)
        # x_checkpoint = self.bn3(x)
        # # print(x.shape)
        # x = self.conv4(x)
        # # print(x.shape)
        # x = self.pool4(x)
        # # print(x.shape)
        # xm = self.bn4(x)

        # x = self.conv5(x)
        # x = self.pool5(x)
        # xm = self.bn5(x)
        # xm = self.bn3_mag(xm)
        # print(xm.shape)

        xm = x_checkpoint.view(
            [x_checkpoint.shape[0], x_checkpoint.shape[1] * x_checkpoint.shape[2] * x_checkpoint.shape[3], 1, 1])
        xm = self.fc_direct(xm)
        # xm = self.fc1(xm)
        # xm = self.relu(self.fc1bn(xm))
        # # xm = self.dropout(xm)
        # xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm

class Net_vanilla_GMUCNN_Hyper(nn.Module):
    def __init__(self, input_channels, layers, kernels, epsilons=[0.0001, 0.0001, 0.0001], hyper_hidden_dim=10, classes=10):
        super(Net_vanilla_GMUCNN_Hyper, self).__init__()

        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons
        self.mode = 'normal'
        self.use_bn = True
        self.padding = [1, 0, 0]
        
        # GMU Layer
        self.gmu1 = GMULayer(input_channels, 64, kernels[0], padding=int((kernels[0] - 1) / 2), 
                             epsilon=epsilons[0], num_slices=2, degree=1, normalize=True)

        # Convolutional Layers
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Pooling layers
        self.mpool1 = nn.AvgPool2d(4)
        self.mpool2 = nn.MaxPool2d(4)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Feature extraction pipeline
        self.feat_net = nn.Sequential(
            self.bn1,
            self.mpool1,
            self.conv2,
            self.bn2,
            nn.ReLU(inplace=True),
            self.mpool2
        )

        # HyperLinearLayer for final classification
        self.hyperlinear = HyperLinearLayer(hyper_input_dim=784, target_input_dim=128, target_output_dim=classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_input):
        # Pass through the GMU layer
        x = x_input
        x = self.gmu1(x, self.training)

        # Feature extraction
        x = self.feat_net(x)

        # Flatten the feature map to prepare for the HyperLinearLayer
        x = x.view(x.shape[0], -1)  # Flatten to shape [batch_size, 784]

        # Use the input itself as the hyper input (you can customize this)
        hyper_input = x

        # Pass through the HyperLinearLayer
        x = self.hyperlinear(hyper_input, x_input)

        return x


class Net_vanilla_GMUCNN(nn.Module):
    def __init__(self,input_channels,layers,kernels,epsilons = [0.0001,0.0001,0.0001],hyper_hidden_dim=10,classes = 10):
        super(Net_vanilla_GMUCNN, self).__init__()
        
        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons 
        self.mode = 'normal'
        self.use_bn = True
        self.padding = [1,0,0]
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
        self.gmu1 = GMULayer(input_channels, 64, kernels[0],padding=int((kernels[0]-1)/2),epsilon = epsilons[0],
                             num_slices=2,degree=1,normalize=True)
        
   
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.mpool1 = nn.AvgPool2d(4)
        self.mpool2 = nn.MaxPool2d(4)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        
        
        self.bnorm_fc = nn.BatchNorm2d(128)
        self.fc1 = nn.Conv2d(128,10,1)
        
        self.feat_net = nn.Sequential(

            # self.conv1,
            self.bn1,
            self.mpool1,
            # nn.ReLU(inplace=True),
            

            self.conv2,
            self.bn2,
            nn.ReLU(inplace=True),
            self.mpool2,
            #
        
        )

        self.drop = nn.Dropout(p=self.dropping)
    
            
        self.relu = nn.ReLU(inplace=True)
        
    
        
    def forward(self, x):
       
        x = self.gmu1(x,net.training)
        x = self.feat_net(x)
                    
       
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        
        xm = self.fc1(xm)

        xm = xm.view(xm.size()[0], xm.size()[1])
        return xm
    
    

class HyperConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, hyper_hidden_dim=10):
        super(HyperConvLayer, self).__init__()
        # Hypernetwork generates weights for convolution
        self.hypernetwork = Hypernetwork(input_channels*784, hyper_hidden_dim,
                                         input_channels * output_channels * kernel_size * kernel_size)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # Aggregate hyper_input to a single vector (e.g., mean of batch or learned representation)
        global_hyper_input = x.mean(dim=0)  # Mean across batch (or customize this step)

        # Generate convolution weights dynamically using the hypernetwork
        weights = self.hypernetwork(global_hyper_input)  # One set of weights for the entire batch
        weights = weights.view(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size)  # Reshape to conv2d dimensions

        # Perform convolution using the generated weights
        conv = nn.functional.conv2d(x, weights, padding=self.kernel_size // 2)  # Convolution with dynamic weights
        return conv



class Net_vanilla_HyperCNN_normal(nn.Module):
    def __init__(self, input_channels, layers, kernels, hyper_hidden_dim=10, classes=10, dropping=0):
        super(Net_vanilla_HyperCNN_normal, self).__init__()
        
        self.layers = layers
        self.dropping = dropping
        
        # Replace GMU with HyperConvLayer
        
        # Normal convolutional layers
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        # Pooling layers
        self.mpool1 = nn.MaxPool2d(4)
        self.mpool2 = nn.MaxPool2d(4)
        self.mpool3 = nn.MaxPool2d(2, padding=1)
        self.mpool4 = nn.MaxPool2d(4)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        # Fully connected layers for classification
        self.bnorm_fc = nn.BatchNorm2d(128)

        # Feature extraction pipeline
        self.feat_net = nn.Sequential(
            self.conv1, 
            self.bn1,
            nn.ReLU(inplace=True),
            self.mpool1,
            self.conv2,
            self.bn2,
            nn.ReLU(inplace=True),
            self.mpool2,
            
        )
        self.hyperlinear = HyperLinearLayer(hyper_input_dim=784, target_input_dim=128, target_output_dim=classes)

        self.drop = nn.Dropout(p=self.dropping)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_input):
        # Pass through the hyperconv layer (weights dynamically generated)
        x = x_input
        x = self.feat_net(x)  # Pass through the feature extraction pipeline
        
        # Flatten and classify
        xm = x.view(x.shape[0], -1)  # Flatten spatial dimensions
        xm = self.hyperlinear(xm, x_input)
        xm = xm.view(xm.size(0), xm.size(1))  # Reshape for output

        return xm



class Net_vanilla_CNN_normal(nn.Module):
    def __init__(self, input_channels, layers, kernels, classes=10, dropping=0):
        super(Net_vanilla_CNN_normal, self).__init__()
        
        self.layers = layers
        self.dropping = dropping
        
        # Replace GMU with HyperConvLayer
        
        # Normal convolutional layers
        self.conv1 = nn.Conv2d(1,64, 3, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        # Pooling layers
        self.mpool1 = nn.MaxPool2d(4)
        self.mpool2 = nn.MaxPool2d(4)
        self.mpool3 = nn.MaxPool2d(2, padding=1)
        self.mpool4 = nn.MaxPool2d(4)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        # Fully connected layers for classification
        self.bnorm_fc = nn.BatchNorm2d(128)
        self.fc1 = nn.Conv2d(128, classes, 1)

        # Feature extraction pipeline
        self.feat_net = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.mpool1,
            self.conv2,
            self.bn2,
            nn.ReLU(inplace=True),
            self.mpool2,
            
        )

        self.drop = nn.Dropout(p=self.dropping)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.feat_net(x)  # Pass through the feature extraction pipeline
        
        # Flatten and classify
        xm = x.view(x.shape[0], -1, 1, 1)  # Flatten spatial dimensions
        xm = self.fc1(xm)
        xm = xm.view(xm.size(0), xm.size(1))  # Reshape for output

        return xm
    
    
def train_network_normal(net, trainloader, testloader, test_labels, init_rate, epochs, weight_decay):
    net = net.cuda()
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=init_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    all_train_losses = []
    train_loss_min = float("inf")

    for epoch in range(epochs):
        train_loss = []
        loss_weights = []

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # Randomly select affine parameters for the current batch
            scale_range = torch.empty(1).uniform_(0.5, 1.0).item()
            augmentation = nn.Sequential(
                K.RandomAffine(degrees=0, translate=None, scale=(scale_range, scale_range), shear=None)
            )

            # Apply random affine transformation
            inputs = augmentation(inputs)

            # Ensure inputs and labels are on the same device as the model
            inputs, labels = inputs.cuda(), labels.cuda()

            # Forward pass
            optimizer.zero_grad()
            allouts = net(inputs)

            # Compute loss and backpropagate
            loss = criterion(allouts, labels.long())
            loss.backward()
            optimizer.step()

            # Log losses
            train_loss.append(loss.item())
            loss_weights.append(len(labels))

        # Compute and store the weighted average loss
        all_train_losses.append(np.average(np.array(train_loss), weights=np.array(loss_weights)))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {all_train_losses[-1]}")

        # Save the best model
        if all_train_losses[-1] < train_loss_min:
            train_loss_min = deepcopy(all_train_losses[-1])
            net_best = deepcopy(net)

    # Evaluate the best model
    net_best = net_best.eval()
    return net_best, all_train_losses




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
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform_test)
        dataset = scale_dataset(dataset,scale)
        dataset_test = scale_dataset(dataset_test,scale)
        
        
        dataset.data = dataset.data.float().unsqueeze(1)
        dataset_test.data = dataset_test.data.float().unsqueeze(1)
    
    if dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.MNIST(root='./data', train=False,
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
    

def test_network(net, testloader, test_labels,input_noise=0):
    net = net.eval()

    correct = torch.tensor(0)
    dataiter = iter(testloader)
    # total = 0 
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            scale_range = torch.empty(1).uniform_(1.0, 1.5).item()
            augmentation = nn.Sequential(
                K.RandomAffine(degrees=0, translate=None, scale=(scale_range, scale_range), shear=None)
            )

            # Apply random affine transformation
            inputs = augmentation(inputs)
            
            inputs = inputs + input_noise*torch.randn_like(inputs)
            all_outs = net(inputs)
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





def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
        
        
        

from corruptions import *
    
def test_network_corruptions(net, testloader, corruptions,atan_convert):
    net = net.eval()
    accuracy_list = [] 
    corruptions = [motion_blur,shot_noise,translate,rotate,shear,scale,
                    spatter,zigzag]
    for corruption in corruptions:
        correct = torch.tensor(0)
        dataiter = iter(testloader)
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs
                inputs, labels = data
                for j in range(inputs.shape[0]):
                    inputs[j,0,:,:] = torch.from_numpy(corruption(255.0*inputs[j,0,:,:].cpu().numpy())).cuda()
                all_outs,temp = net(inputs)
                
                predicted = torch.argmax(all_outs,1)

                correct = correct + torch.sum(predicted == labels)
        accuracy = float(correct) / float(10000)
        accuracy_list.append(accuracy)
        print("Corruption:",corruption, " Accuracy: ", accuracy)
    
    print("Mean accuracy:",np.mean(accuracy_list))
    return 0




if __name__ == "__main__":
    
    
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    corruptions = ['brightness','canny_edges','dotted_line','fog','glass_blur','identity',
                    'impulse_noise','motion_blur','shot_noise','translate','rotate','shear','scale',
                    'spatter','zigzag']
    corruptions = ['general']
    atanh_convert = 0 

    gc.collect()
    torch.cuda.empty_cache()
    
    dataset_name = "MNIST"
    batch_size = 100
    init_rate = 0.005
    init_rate_crank = 0.01
    labelnoise = 0
    input_noise = 0 
    input_noise_array = [0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.25,0.3,0.35]
    epsilons = [ .01]
    gmu_epsilons = [0.0001,0.00001,0]
    hyper_hidden_dim = 10
    # test_transforms = ['scale','translate','rotate','']
    
    
    step_size = 10
    gamma_learning = 0.8
    total_epoch = 200
    total_epoch_crank = 100
    decay_normal = 0    
    decay_regress = 0
    decay_errors = 0
    dropping = 0 
    
    decay_normal_crank = 0 
    
    layers = [25,50]
    kernels = [3,3]
    layers_crank = []
    
    training_size = 1000
    mode = 'regress_batch'
    use_bn = True
    rescale = 1.0
    rank_convert = False
    input_channels = 1
    global DEGREES
    DEGREE = [3,8]
    alpha=0.2
    net = Net_vanilla_HyperCNN_normal(input_channels, layers, kernels, hyper_hidden_dim=hyper_hidden_dim)
    # net = Net_vanilla_CNN_normal(input_channels, layers, kernels)
    # net = Net_vanilla_GMUCNN_Hyper(input_channels, layers, kernels, hyper_hidden_dim=hyper_hidden_dim)
    # net = Net_vanilla_GMUCNN(input_channels,layers,kernels,gmu_epsilons,decay_regress,decay_errors)
    
    dataset,dataset_test,my_dataset,my_dataset_test,trainloader,testloader = load_data_and_generators(dataset_name,training_size,rescale,rank_convert,atanh_convert,labelnoise)
    
    net.mode = 'normal'
    
    net,all_losses = train_network_normal(net,trainloader,testloader,my_dataset_test.labels, init_rate,total_epoch,decay_normal)
    # net.load_state_dict(torch.load('CNN_convert_3slice_1degree_FashionMNIST_[3, 8]degree_300epochs_[25, 50]layers60000data.h5'))
    
    accuracy = test_network(net, testloader, my_dataset_test.labels, 0)
    print('Test Accuracy (Noise=',input_noise,"):", accuracy)
    # test_network_corruptions(net, testloader, corruptions,atanh_convert)
    
    
    


    
    
