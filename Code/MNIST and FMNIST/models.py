# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:56:29 2026

@author: User
"""
from torch import nn
import torch
import numpy as np

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
        
        n = self.input_channels*self.output_channels
        stdv = 1. / np.sqrt(n)
        self.weights.data.uniform_(-stdv, stdv)
        # torch.nn.init.xavier_normal_(self.weights,gain=0.01)
        self.weight_bias.data.uniform_(0, 0.5)
        
        
    def forward(self, y2,train_status=True):

        y = nn.Unfold((self.weights.shape[2],self.weights.shape[3]),padding=self.padding)(y2)        
        y = y + self.epsilon*torch.randn_like(y)
        # y = torch.hstack([y,torch.ones(y.shape[0],1,y.shape[2]),torch.zeros(y.shape[0],1,y.shape[2])])
        
        
    
        # if self.num_slices == 0:
        #     y= y - y.mean(dim=1).unsqueeze(1).repeat(1,y.shape[1],1)
        #     GG = torch.std(y,dim=1)  
        #     y = y/GG.unsqueeze(1).repeat(1,y.shape[1],1)
        #     y = y.unsqueeze(1).repeat(1,self.output_channels,1,1)
        #     X = self.weight_bias
        #     X = X.unsqueeze(0)
        #     X = X.view(X.shape[0],X.shape[1],X.shape[3]*X.shape[4],1)
        #     y = y - (X.repeat(y.shape[0],1,1,y.shape[3]))
        #     err = torch.mean((y)**2,dim=2)
        #     err = err.view(err.shape[0],err.shape[1],int(np.sqrt(err.shape[2])),int(np.sqrt(err.shape[2])))
        #     return (torch.exp(-err))
        
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

        # self.drop = nn.Dropout(p=0.2)
    
            
        self.relu = nn.ReLU(inplace=True)
        
    
        
    def forward(self, x):
       
        
        x = self.feat_net(x)                    
       
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        # xm = self.drop(xm)
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

        self.drop = nn.Dropout(p=self.dropping)
    
            
        self.relu = nn.ReLU(inplace=True)
        
    
        
    def forward(self, x):
       
        if self.mode == 'normal':
            x = self.gmu1(x,True)
            x = self.feat_net(x)
            x_errs = x
                    
       
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        
        xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)

        xm = xm.view(xm.size()[0], xm.size()[1])
        return xm,x_errs
    