# -*- coding: utf-8 -*-

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
# from corruptions import *
from PIL import Image
import matplotlib.pyplot as plt

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

from matplotlib.widgets import Button
    
label_dict = {0: "T-shirt",
             1: "Trouser",
             2: "Pullover",
             3: "Dress",
             4: "Coat", 
             5: "Sandal", 
             6: "Shirt",
             7: "Sneaker",
             8: "Bag",
             9: "Ankle Boot"}







def SRModule_Conv(X,y2,p, padding,decay_regress, epsilon=0.0001, normalize=True,exponent=False):
    
    
    p=3
    y = nn.Unfold((X.shape[2],X.shape[3]),padding=padding)(y2)
    y = y + epsilon*torch.randn_like(y)
    if normalize:
        GG = torch.std(y,dim=1)  
        y = y/GG.unsqueeze(1).repeat(1,y.shape[1],1)
    
    N = X.shape[0]
    if len(y.shape)==1:
        y = y.unsqueeze(0)
    
    X = X.view(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3])
    X = X.unsqueeze(2)
    X = torch.stack([torch.ones(X.shape), X],dim=3).squeeze()
    for i in range(p-1):
        X = torch.concat((X, (X[:,:,1]**(i+2)).unsqueeze(2)),dim=2)
        
    X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
    X_cov_inv = torch.linalg.inv(X_cov+ decay_regress)
    M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
    W = torch.einsum('ijk,akb->aijb',M,y)
    
    
    pred_final = torch.einsum('bec,abcd->abed', X, W)
    
    
    err = torch.mean((y.unsqueeze(1).repeat(1,pred_final.shape[1],1,1)-pred_final)**2,dim=2)
    
    err = err.view(err.shape[0],err.shape[1],int(np.sqrt(err.shape[2])),int(np.sqrt(err.shape[2])))
    # err = err/err.detach().max()
    if exponent:
        # print('here')
        A = (torch.exp(-err)-np.exp(-1.0))/(np.exp(0)-np.exp(-1.0))
        # return torch.exp(-err)
        return A-0.5
    else:
        if normalize:
            return 1-err
        else:
            return -err
    
    
    



from torcheval.metrics import R2Score




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




    

class Net_vanilla_CNN_normal(nn.Module):
    def __init__(self,input_channels,layers,kernels,epsilons = [0.0001,0.0001,0.0001],decay_regress=0,decay_errors = 0, classes = 10,use_bn = True,dropping=0,poly_order_init=5):
        super(Net_vanilla_CNN_normal, self).__init__()
        
        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons 
        self.mode = 'normal'
        self.use_bn = True
        self.poly_order_init = poly_order_init
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
        
        
        
        if self.mode =='normal':
            self.conv1 = nn.Conv2d(1,64,3, padding=1)
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
                self.bn1,
                nn.ReLU(inplace=True),
                self.mpool1,
    
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
        
    def conv_regress_batch_standalone(self,weights,inputs,p,epsilon, padding = 0, normalize=True, exponent = True):
        
        layer_out = SRModule_Conv(weights,inputs,p,padding, self.decay_regress,epsilon,  normalize=normalize,exponent=exponent)
        return layer_out
        
    
        
    def forward(self, x,bazinga=1):
        
        
        if self.mode == 'normal':
            
            x = self.feat_net(x)
            x_errs = x
                    
                
       
            
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
       
        xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)
        
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm,x_errs

import itertools as it
class SRNLayer(nn.Module):
    def __init__(self,input_channels, output_channels, kernel_size, padding = 0, epsilon = 0.0001, num_slices=2,degree=4,exponent=True, normalize = True):
        super(SRNLayer, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,kernel_size,kernel_size,num_slices))
        torch.nn.init.kaiming_uniform_(self.weights)
        self.exponent = exponent
        self.normalize = normalize
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.degree = degree 
        self.epsilon = epsilon
        self.padding = padding
        self.iter =  it.combinations(np.arange(num_slices), 2)

    def forward(self, y2):
        # print(self.weights.shape)
        y = nn.Unfold((self.weights.shape[2],self.weights.shape[3]),padding=self.padding)(y2)
        # print(self.epsilon)
        y = y + self.epsilon*torch.randn_like(y)
        if self.normalize:
            GG = torch.std(y,dim=1)  
            # A =(GG<epsilon).float()*epsilon + (GG>=epsilon).float()*torch.std(y,dim=1)
            y = y/GG.unsqueeze(1).repeat(1,y.shape[1],1)
            
        X = self.weights 
        X = X.view(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3],self.num_slices)
        
        
        for i in range(self.degree-1):
            # X = torch.concat((X, (torch.exp(-alpha((i+2)*X[:,:,1])).unsqueeze(2)),dim=2)
            X = torch.concat((X, X[:,:,0:self.num_slices]**(i+2)),dim=2)
            # X = torch.concat((X, (X[:,:,1]**(i+2)).unsqueeze(2)),dim=2)
        
        # for i,j in self.iter:
        #     # X = torch.concat((X, (torch.exp(-alpha((i+2)*X[:,:,1])).unsqueeze(2)),dim=2)
        #     if i!=j:
        #         X = torch.concat((X, (self.weights[:,:,i]*self.weights[:,:,j]).unsqueeze(2)),dim=2)
            # X = torch.concat((X, (X[:,:,1]**(i+2)).unsqueeze(2)),dim=2)
        
        X = torch.concat((torch.ones((X.shape[0],X.shape[1],1),requires_grad=False), X),dim=2)
        
        X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
        # X_cov = torch.einsum('abdc,abce->abde', X.permute(0,1,3,2),X)
        
        # print(decay_regress)
        # X_cov_inv = torch.linalg.inv(X_cov+ decay_regress)
        X_cov_inv = torch.linalg.inv(X_cov)
        M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
        W = torch.einsum('ijk,akb->aijb',M,y)
        pred_final = torch.einsum('bec,abcd->abed', X, W)
        
        
        
        err = torch.mean((y.unsqueeze(1).repeat(1,pred_final.shape[1],1,1)-pred_final)**2,dim=2)
        
        err = err.view(err.shape[0],err.shape[1],int(np.sqrt(err.shape[2])),int(np.sqrt(err.shape[2])))
        # err = err/err.detach().max()
        if self.exponent:
            # print('here')
            A = (torch.exp(-err)-np.exp(-1.0))/(np.exp(0)-np.exp(-1.0))
            # return torch.exp(-err)
            return A-0.5
        else:
            if self.normalize:
                return 1-err
            else:
                return -err
            
   

    
    
class Net_vanilla_CNN_convert(nn.Module):
    def __init__(self,input_channels,layers,kernels,epsilons = [0.0001,0.0001,0.0001],decay_regress=0,decay_errors = 0, classes = 10,use_bn = True,dropping=0,poly_order_init=5):
        super(Net_vanilla_CNN_convert, self).__init__()
        
        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons 
        self.mode = 'normal'
        self.use_bn = True
        self.poly_order_init = poly_order_init
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
                
                # self.conv1,
                # self.bn1,
                # nn.ReLU(inplace=True),
                # self.mpool1,
    
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
        
    
    def conv_regress_batch_standalone(self,weights,inputs,p,epsilon, padding = 0, normalize=True, exponent = True):
        
        layer_out = SRModule_Conv(weights,inputs,p,padding, self.decay_regress,epsilon,  normalize=normalize,exponent=exponent)
        return layer_out
        
    
        
    def forward(self, x,bazinga=1):
       
        # if self.mode == 'normal':
            
        x = self.conv_regress_batch_standalone(self.conv1.weight,
                                       x,epsilon = self.epsilons[0],padding = self.padding[0], p=3,exponent=True)
        
        x = self.feat_net(x)
        x_errs = x
            
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        
        xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)
       
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm,x_errs


class Net_vanilla_CNN_convert_old(nn.Module):
    def __init__(self,input_channels,layers,kernels,epsilons = [0.0001,0.0001,0.0001],decay_regress=0,decay_errors = 0, classes = 10,use_bn = True,dropping=0,poly_order_init=5):
        super(Net_vanilla_CNN_convert_old, self).__init__()
        
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
        
        # ----------------------- Old code----------------------------
        # if self.mode =='regress_batch':
        self.conv1 = nn.Conv2d(input_channels, layers[0], kernels[0])
        self.pool1 = nn.AvgPool2d(kernel_size=(3,3))
        self.pool2 = nn.AvgPool2d(kernel_size=(2,2))
        
        self.conv2 = nn.Conv2d( layers[0], layers[1], kernels[1]) 
        
        
        
        self.bn1 = nn.BatchNorm2d(layers[0])
        self.bn2 = nn.BatchNorm2d(layers[0],affine=False)
        self.bn3 = nn.BatchNorm2d(450)
        self.fc_direct = nn.Conv2d(450, classes, 1)

        # ----------------------- Old code----------------------------
        
        

        self.drop = nn.Dropout(p=self.dropping)
        # print(self.dropping)
    
            
        self.relu = nn.ReLU(inplace=True)
        # self.fc_direct0 = nn.Conv2d(layers[0], 200, 1)
        # self.bn0 = nn.BatchNorm2d(200)
        # self.fc_direct = nn.Conv2d(1250, classes, 1)
    
            
    # def regress_batch_layer(self,weights,inputs,Mul_mats):
    #     # with torch.no_grad():
    #     layer_out = multi_regress_withgrad(weights,inputs,DEGREE,Mul_mats,normalize=True,exponent=False)
    #     return layer_out.unsqueeze(2).unsqueeze(3)
    
    def conv_regress_batch_standalone(self,weights,inputs,p,epsilon, padding = 0, normalize=True, exponent = True):
        # with torch.no_grad():
        # print(weights[0][0])
        # print(padding)
        layer_out = SRModule_Conv(weights,inputs,p,padding, self.decay_regress,epsilon,  normalize=normalize,exponent=exponent)
        return layer_out
        
    
        
    def forward(self, x,bazinga=1):
        # print(x.shape)
        # x = x.reshape(x.shape[0],1,28,28)
        # x = GaussianLayer()(x)
        # x = x.reshape(x.shape[0],784,1,1)
        # x = x*(self.hada_mult1.repeat(x.shape[0],1,1,1))
        
        if self.mode == 'normal':
            # for i in range(len(self.convs)):
            #     x = self.convs[i](x)
            #     # x = self.drop(x)
            #     x = self.relu(x)
            #     x = self.bns[i](x)
            # print('baby step outside')
            x = self.conv_regress_batch_standalone(self.conv1.weight,
                                           x,epsilon = self.epsilons[0],padding = self.padding[0], p=DEGREE[0],exponent=True)
            
            x = self.feat_net(x)
            x_errs = x
                    
                
        elif self.mode =='just_normal':
            
            # for i in range(len(self.convs)):
                # x = self.convs[i](x)
                # # x = self.drop(x)
                # x = self.relu(x)
            x = self.conv1(x)
            x = self.relu(x)
        
                # print(torch.mean(x))
        elif self.mode == 'regress_batch':
            # for i in range(len(self.convs)):
                # T = time.time()
                # x = self.regress_batch_layer(self.convs[i].weight.squeeze(),
                                             # x.squeeze(),self.Mul_mats[i])
                                             
            # print('baby step inside')
            # print(x.shape)
            x = self.conv_regress_batch_standalone(self.conv1.weight,
                                         x,epsilon = self.epsilons[0],padding = self.padding[0], p=DEGREE[0],exponent=True)
            # print(x.shape)
            # print(self.padding[0])
            # print(x_errs.shape)
            x = self.pool1(x)
            # print(x.shape)
            # if self.use_bn:
            x = self.bn2(x)
            # print(x.shape)
            x_errs = self.conv_regress_batch_standalone(self.conv2.weight,
                                         x,epsilon = self.epsilons[1],padding = self.padding[1], p=DEGREE[1],exponent=True)
            # print(x.shape)
            x = self.pool2(x_errs)
            
            # x = self.relu(self.bn3(x))
            # print(x.shape)
            
            # print(x.shape)
                # x = self.bias1(x)
            # else:
            #     x = x_errs
            x = self.drop(x)
            
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        xm = self.bn3(xm)
                # print("Time:",time.time()-T)
                
                
        # else:
        #     # for i in range(len(self.convs)):
        #     x = self.CRank_layer(x,self.conv1,0)
        #     # x_errs = x
        #     x_errs = x 
        
            
                
        
                # x = BiasLayer(x.shape)(x)
                # x = self.relu(x)
                # x = self.bns_rank[i](x)
        # elif self.mode == 'chatterjee_normal':

        #     for i in range(len(self.convs)):
        #         x = self.CRank_layer(x,self.convs[i])
                
        #         # x = self.bns_rank[i](x)
            
        # elif self.mode == 'chatterjee_norank':
        #     for i in range(len(self.convs)):
        #         x = self.CRank_layer(x,self.convs[i])
        #         # x = BiasLayer(x.shape)(x)
        #         # x = self.relu(x)
        #         x = self.bns_rank[i](x)
                
        # elif self.mode == 'spearman':
        #     for i in range(len(self.convs)):
        #         x = self.CRank_layer(x,self.convs[i])
        #         # x = BiasLayer(x.shape)(x)
        #         # x = self.relu(x)
        #         # x = self.bns_rank[i](x)
            
        # x = self.relu(self.fc_another(x)
        # x = self.relu(self.fc_direct0(x))
        # x = self.bn0(x)
        # if bazinga == 0 or bazinga == 1:
        xm = self.fc_direct(xm)
        # print(xm.shape)
        # xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        # xm = self.fc2(xm)
        
        # else:
        #     # print('g')
        #     xm = self.regress_batch_standalone(self.fc_direct.weight.squeeze(),
        #                                   x.squeeze(),p=1,normalize=True,exponent=True)
        #     xm = self.final_bn(xm)
        
        # xm = self.fc1(xm)
        # xm = self.relu(self.fc1bn(xm))
        # # xm = self.dropout(xm)
        # xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm,x_errs
    
    
class Net_vanilla_NN_crank(nn.Module):
    def __init__(self,input_channels,layers,decay_regress=0.001,decay_errors = 0.001, classes = 10,use_bn = True):
        super(Net_vanilla_NN_crank, self).__init__()
        
        self.layers = layers
        self.decay_errors = 0 
        self.decay_regress = 0
        
        # network layers
        self.convs = []
        self.bns = []
        self.bns_rank = [] 
        self.first_bn = nn.BatchNorm2d(input_channels)
        
        # if len(layers)>0:
        #     self.convs.append(nn.Conv2d(input_channels, layers[0], 1))
        #     if use_bn == True:
        #         self.bns.append(nn.BatchNorm2d(layers[0]))  
        #         self.bns_rank.append(nn.BatchNorm2d(layers[0]))  
    
        #     else:
        #         self.bns.append(nn.Identity())
            
        #     for i in range(len(self.layers)-1):
        #         self.convs.append(nn.Conv2d(layers[i],layers[i+1], 1))
        #         if use_bn == True:
        #             self.bns.append(nn.BatchNorm2d(layers[i+1])) 
        #             self.bns_rank.append(nn.BatchNorm2d(layers[i+1]))  
        #         else:
        #             self.bns.append(nn.Identity())
        #     self.fc_direct = nn.Conv2d(layers[-1], classes, 1)
        # else:
        self.fc_direct = nn.Conv2d(input_channels, classes, 1)

            
        self.relu = nn.ReLU()
        # self.fc_another = nn.Conv2d(layers[-1],layers[-1], classes, 1)
    
        
        
        
        
        
    def forward(self, x):
        # print(x.shape)
        # x = x.reshape(x.shape[0],1,28,28)
        # x = GaussianLayer()(x)
        # x = x.reshape(x.shape[0],784,1,1)
        # x = self.first_bn(x)
        
        # for i in range(len(self.layers)):
        #     x = self.convs[i](x)
        #     x = self.relu(x)
        #     x = self.bns[i](x)
       
        xm = self.fc_direct(x)
        # xm = self.fc1(xm)
        # xm = self.relu(self.fc1bn(xm))
        # # xm = self.dropout(xm)
        # xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm,torch.zeros(1)
    

def train_network_convert(net,trainloader, rate_array,epoch_array,weight_decay_array,mode):
    net = net
    net = net.cuda()
    net = net.train()
#     optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=rate_array[0], weight_decay=weight_decay_array[0])

#     scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma_learning)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], last_epoch=-1)

    criterion = nn.CrossEntropyLoss()
    
    init_epoch = 0
    all_train_losses = []
    net.mode = 'normal'
    # bazinga = 0 
    for epoch in range(epoch_array[0]):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # scheduler.step()
        # print('epoch: ' + str(epoch))
        train_loss = []
        loss_weights = [] 
        # if epoch>= 
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # inputs = inputs.cuda()
            # labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            allouts,temp = net(inputs)

            loss = criterion(allouts, labels.long())
            loss.backward()
            train_loss.append(loss.item())
            loss_weights.append(len(labels))
            
            
            optimizer.step()
            # print(0)
        
        all_train_losses.append(np.average(np.array(train_loss),weights=np.array(loss_weights)))
    
    accuracy = test_network(net, testloader, my_dataset_test.labels)
    train_accuracy = test_network(net, trainloader, my_dataset.labels)
    
    print('Train Accuracy:', train_accuracy)
    print('Test Accuracy:', accuracy)
    
    net = net.train()
    for g in optimizer.param_groups:
        g['lr'] = rate_array[1]
        g['weight_decay'] = weight_decay_array[1]

    for epoch in range(epoch_array[1]):
        net.mode = mode
        train_loss = []
        loss_weights = [] 
        print('Round 2:', epoch)
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # inputs = inputs.cuda()
            # labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            allouts,temp = net(inputs)

            loss = criterion(allouts, labels.long())
            loss.backward()
            train_loss.append(loss.item())
            loss_weights.append(len(labels))
            
            
            optimizer.step()
            # print(0)
        
        all_train_losses.append(np.average(np.array(train_loss),weights=np.array(loss_weights)))
        print(all_train_losses[-1])

        # print('break')
        
    # train_loss = train_loss/len(trainloader.sampler)
#     train_loss = np.mean(all_train_losses[-5:])
    # train_loss = all_train_losses[-1]
    net = net.eval()
    feat_shape = temp.shape

    return net,all_train_losses,feat_shape





def predict_sample(net, inputs, samples=20, mode='max', min_scale=0.6):
    net = net.eval()
    inputs = inputs.repeat(samples, 1, 1, 1)
    correct = torch.tensor(0)
    
    # Split samples into two halves
    half_samples = samples // 2
    random_crop_aug = kornia.augmentation.RandomResizedCrop((28, 28), scale=(min_scale, 1.0), 
                                          ratio=(3.0 / 4.0, 4.0 / 3.0), resample="BILINEAR", same_on_batch=False)
    horizontal_flip_aug = kornia.augmentation.RandomHorizontalFlip(p=1.0)  # Mirror flip
    
    with torch.no_grad():
        # Apply random crop to the first half
        inputs = random_crop_aug(inputs)  # Apply random crop
        inputs = horizontal_flip_aug(inputs)
        
        probs = nn.Softmax(dim=1)
        all_outs, temp = net(inputs)
        predicted = torch.argmax(all_outs, 1)
        all_probs = probs(all_outs)
        
        if mode == 'max':
            index = torch.argmax(all_probs.topk(1, dim=1).values)
            return predicted[index], 0
        elif mode == 'weighted':
            belief = torch.mean(all_probs, 0)
            predicted = torch.argmax(belief)
            return predicted, belief
        elif mode == 'voting':
            vals = torch.mode(predicted).values
            return vals, 0



import cv2


class Draw_preds:
    def __init__(self,text1,text2,text3):
        super(Draw_preds, self).__init__()
        self.text1 = text1
        self.text2 = text2
        self.text3 = text3 
        dataset_name = "FashionMNIST"
        batch_size = 200
        init_rate = 0.0005
        init_rate_crank = 0.01
        labelnoise = 0
        input_noise = 0 
        input_noise_array = [0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.25,0.3,0.35]
        epsilons = [ .05]
        srn_epsilons = [0.0001,0.00001,0]
        
        step_size = 10
        gamma_learning = 0.8
        total_epoch = 400
        total_epoch_crank = 100
        decay_normal = 0  
        decay_regress = 0
        decay_errors = 0
        dropping = 0 
        
        decay_normal_crank = 0 
        
        layers = [25,50]
        kernels = [5,3]
        layers_crank = []
        
        training_size = 60000
        mode = 'regress_batch'
        use_bn = True
        rescale = 1.0
        rank_convert = False
        input_channels = 1
        global DEGREES
        DEGREE = [1,8]
        alpha=0.2

        self.net2 = Net_vanilla_CNN_convert(input_channels,layers,kernels,srn_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)
        self.net = Net_vanilla_CNN_normal(input_channels,layers,kernels,srn_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)
        
        
        
        self.net1 = Net_vanilla_CNN_convert(input_channels,layers,kernels,srn_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)
        
        
        self.net1.load_state_dict(torch.load('Best_CNN_convert_FashionMNIST_[3, 8]degree_400epochs_[25, 50]layers60000data.h5',weights_only=True))
        self.net.load_state_dict(torch.load('CNN_normal_FashionMNIST_[3, 8]degree_400epochs_[25, 50]layers60000data.h5',weights_only=True))

        self.net2 = self.net1 
        
        self.net1 = self.net1.eval()
        self.net = self.net.eval()
        self.net2 = self.net2.eval()
        # self.net3 = self.net3.eval()
        
        # print('loaded')
    
    
    
    def next(self, event):
        # plt.text ()s
        # print('no')
        Im = Image.open('to_predict_fmnist.png').convert('L').resize((28,28),Image.BILINEAR)
        Im = torch.from_numpy(np.asarray(Im)).float()/255.0
        
        
        ax.imshow(Im)
        fig.canvas.draw_idle()
        # plt.imshow(Im)
        # self.fig.plot(Im)
        
        # Im = Im.unsqueeze(0).unsqueeze(0).repeat(2,1,1,1)
        out_srn,b1 = predict_sample(self.net1,Im.cuda(),20,'weighted',min_scale=0.8)
        out_srn2,b2 = predict_sample(self.net2,Im.cuda(),20,'weighted',min_scale=0.8)
        # out_srn3,b3 = predict_sample(self.net3,Im.cuda(),20,'max',min_scale=0.8)
        
        out_normal,ll = predict_sample(self.net,Im.cuda(),20,'weighted',min_scale=0.8)
        Im = 1- Im
        out_normal_inv,ll = predict_sample(self.net,Im.cuda(),20,'weighted',min_scale=0.8)
        
        self.text1.set_text('CNN-I:'+str(label_dict[int(out_normal_inv.cpu().numpy())]))
        self.text2.set_text('GMU:'+str(label_dict[int(out_srn.cpu().numpy())]))
        self.text3.set_text('CNN:'+str(label_dict[int(out_normal.cpu().numpy())]))
        fig.canvas.draw_idle()
        
        
    

if __name__ == "__main__":
    plt.close('all')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    
    # corruptions = [brightness,canny_edges,dotted_line,fog,glass_blur,identity,
    #                 impulse_noise,motion_blur,shot_noise,translate,rotate,shear,scale,
    #                 spatter,zigzag]
    # corruptions = ['general']
    atanh_convert = 0 

    # torch.cuda.empty_cache()
    gc.collect()
    # gc.collect()
    
    torch.cuda.empty_cache()
    
    dataset_name = "FashionMNIST"
    batch_size = 200
    init_rate = 0.0005
    init_rate_crank = 0.01
    labelnoise = 0
    input_noise = 0 
    input_noise_array = [0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.25,0.3,0.35]
    epsilons = [ .05]
    srn_epsilons = [0.0001,0.00001,0]
    
    step_size = 10
    gamma_learning = 0.8
    total_epoch = 400
    total_epoch_crank = 100
    decay_normal = 0  
    decay_regress = 0
    decay_errors = 0
    dropping = 0 
    
    decay_normal_crank = 0 
    
    layers = [25,50]
    kernels = [5,3]
    layers_crank = []
    
    training_size = 60000
    mode = 'regress_batch'
    use_bn = True
    rescale = 1.0
    rank_convert = False
    input_channels = 1
    global DEGREES
    # DEGREE = [4,8]
    alpha=0.2
    
    
    
    # test_transforms = ['scale','translate','rotate','']
    # dataset,dataset_test,my_dataset,my_dataset_test,trainloader,testloader,testloader_adversary = load_data_and_generators(dataset_name,training_size,rescale,rank_convert,atanh_convert,labelnoise)

    # net1 = Net_vanilla_CNN_convert_5kernel(input_channels,layers,kernels,srn_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)
    # net2 = Net_vanilla_CNN_convert(input_channels,layers,kernels,srn_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)
    # net = Net_vanilla_CNN_normal(input_channels,layers,kernels,srn_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)
    # net3 = Net_vanilla_CNN_convert_old(input_channels,layers,kernels,srn_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)
    # net3.mode = 'regress_batch'
    # # net1.load_state_dict(torch.load('Best_CNN_convert_FashionMNIST_[3, 8]degree_400epochs_[25, 50]layers60000data.h5',weights_only=True))
    # # net1.load_state_dict(torch.load('CNN_convert_augmented_MNIST_[3, 8]degree_200epochs_[25, 50]layers60000data.h5',weights_only=True))
    # # net1.load_state_dict(torch.load('CNN_convert_5kernel_MNIST_[3, 8]degree_200epochs_[25, 50]layers60000data.h5',weights_only=True))
    # # net1.load_state_dict(torch.load('CNN_convert_5kernel_withaug_MNIST_[4, 8]degree_500epochs_[25, 50]layers60000data.h5',weights_only=True))
    # net1.load_state_dict(torch.load('CNN_convert_5kernel_noaug_lessepsilon_MNIST_[3, 8]degree_300epochs_[25, 50]layers60000data.h5',weights_only=True))
    # net2.load_state_dict(torch.load('CNN_convert_MNIST_[3, 8]degree_200epochs_[25, 50]layers60000data.h5',weights_only=True))
    # net3.load_state_dict(torch.load('C_SRM_MNIST_[4, 8]degree_200epochs_[25, 50]layers60000data.h5',weights_only=True))
    
    # net1 = net1.eval()
    # net = net.eval()
    # net2 = net2.eval()
    # net3 = net3.eval()
    
    # # accuracy = test_network(net3, testloader, my_dataset_test.labels, 0)
    # # print('Test Accuracy (Noise=',input_noise,"):", accuracy)
    
    # # test_network_corrupted(net3, dataset_name, corruptions,atanh_convert)
    

    # # net.load_state_dict(torch.load('CNN_normal_FashionMNIST_[3, 8]degree_400epochs_[25, 50]layers60000data.h5',weights_only=True))
    # net.load_state_dict(torch.load('CNN_normal_MNIST_[3, 8]degree_200epochs_[25, 50]layers60000data.h5',weights_only=True))

    

    # probs = nn.Softmax(dim=1)
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    fig.subplots_adjust(bottom=0.2)
    
    
    
    text1 = fig.text(0,0.9,'CNN-I:',fontsize=30)
    text2 = fig.text(0.4,0.9,'GMU:',fontsize=30)
    text3 = fig.text(0.75,0.9,'CNN:',fontsize=30)
    # plt.tight_layout()
    callback = Draw_preds(text1,text2,text3)
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Update')
    
    
    bnext.on_clicked(callback.next)
    
    plt.show()

    
        
    
    # print('\n')
    
    # print('Next Label (original):', predicted[0][1], " Probability:", prob[0,predicted[0][1]])
    # print('Next Label (srn):', predicted1[0][1], " Probability:", prob1[0,predicted1[0][1]])      
    
    
    # net.load_state_dict(torch.load(''))
    # print('Train Accuracy:', train_accuracy)
    # accuracy = test_network(net, testloader, my_dataset_test.labels, 0)
    # print('Test Accuracy (Noise=',input_noise,"):", accuracy)
    
    # test_network_corrupted(net, dataset_name, corruptions,atanh_convert)
    
    # for epsilon in epsilons:
    #     final_acc, adv_examples = test_network_with_adversary( net, testloader_adversary, epsilon= epsilon)
    #     print('Test Accuracy (Epsilon=',epsilon,"):", final_acc)
        
    # # accuracy = test_network_dilate(net, testloader, my_dataset_test.labels, 0)
    # print('Test Accuracy (Dilated):', accuracy)
    
    
    # net.mode = 'normalized_smoothness'
    # net_crank = Net_vanilla_NN_crank(feat_shape[1],layers_crank,decay_regress,decay_errors,use_bn = use_bn)
    # # print(feat_shape)
    # crank_dataset,crank_dataset_test,crank_trainloader,crank_testloader = create_crank_generators(net, dataset,dataset_test, feat_shape,batch_size=400)
    
    # print('Crank Data Ready')
    
    # net_crank,all_losses,feat_shape = train_network_normal(net_crank,crank_trainloader, init_rate_crank,total_epoch_crank,decay_normal_crank)
    # accuracy = test_network(net_crank, crank_testloader, crank_dataset_test.labels)
    # train_accuracy = test_network(net_crank, crank_trainloader, crank_dataset.labels)
    
    # print('Train Accuracy (crank):', train_accuracy)
    # print('Test Accuracy (crank):', accuracy)
    
    # a = input('')
    
    
    # accuracy = test_network_dilate(net_normal, testloader, my_dataset_test.labels,input_noise)
    # print('Test Accuracy (Dilation)', accuracy)
    
    # for input_noise in input_noise_array:
        # accuracy = test_network(net, testloader, my_dataset_test.labels,input_noise)
        # print('Test Accuracy (Noise=',input_noise,"):", accuracy)
    
        
        
    # layers = [400]
    
    # dropping = 0 
    # net_normal = Net_vanilla_CNN_convert(input_channels,layers,use_bn=use_bn,dropping=dropping)
    # net_normal.mode = 'normal'
    
    # # total_epoch = 100
    # init_rate = 0.0005
    # net_normal,all_losses,feat_shape = train_network_normal(net_normal,trainloader, init_rate,total_epoch,decay_normal)
    # train_accuracy = test_network(net_normal, trainloader, my_dataset.labels)
    
    # print('Train Accuracy:', train_accuracy)
    # accuracy = test_network(net_normal, testloader, my_dataset_test.labels, 0)
    # print('Test Accuracy (Noise=',input_noise,"):", accuracy)
    
    # test_network_corrupted(net_normal, dataset_name, corruptions,atanh_convert)

    # # accuracy = test_network_dilate(net_normal, testloader, my_dataset_test.labels, 0)
    # # print('Test Accuracy (Dilated):', accuracy)
    
    # # accuracy = test_network_dilate(net_normal, testloader, my_dataset_test.labels,input_noise)
    # # print('Test Accuracy (Dilation)', accuracy)
    
    
    # # for input_noise in input_noise_array:
    # #     accuracy = test_network(net_normal, testloader, my_dataset_test.labels,input_noise)
    # #     print('Test Accuracy (Noise=',input_noise,"):", accuracy)
    
    # for epsilon in epsilons:
    #     final_acc, adv_examples = test_network_with_adversary( net_normal, testloader_adversary, epsilon= epsilon)
    #     print('Test Accuracy (Epsilon=',epsilon,"):", final_acc)
        
        
        
        
    
    
    
    
    # net.mode = mode
    # net_crank = Net_vanilla_NN_crank(feat_shape[1], layers_crank)
    
    # crank_dataset,crank_dataset_test,crank_trainloader,crank_testloader = create_crank_generators(net, dataset,dataset_test, feat_shape,batch_size=400)
    
    # print('Crank Data Ready')
    
    # net_crank,all_losses,feat_shape = train_network_normal(net_crank,crank_trainloader, init_rate_crank,total_epoch_crank,decay_normal_crank)
    # accuracy = test_network(net_crank, crank_testloader, crank_dataset_test.labels)
    # train_accuracy = test_network(net_crank, crank_trainloader, crank_dataset.labels)
    
    # print('Train Accuracy (crank):', train_accuracy)
    # print('Test Accuracy (crank):', accuracy)
    
    
