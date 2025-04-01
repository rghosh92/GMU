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
# import kornia
from scipy import stats


import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
from data_loaders import *

# torch.set_float32_matmul_precision("high")

# import torchsort
# def get_rank(x):
#    rank_tensor = torch.zeros_like(x)
#    for i in range(len(rank_tensor)):
#        rank_tensor[i] = (x[i]>x).long().sum()
#    return rank_tensor
# import time 

# x = torch.randn(10000)

# T = time.time()
# rank1 = get_rank(x)
# print("T1:",time.time()-T)

# T = time.time()
# rank2 = get_rank_argsort(x)
# print("T2:",time.time()-T)

# print("Check:",torch.sum((rank1-rank2)**2))

# a = input('')

def get_rank_argsort(x):
    indices = torch.argsort(x)
    rank_tensor = torch.argsort(indices)
    # rank_tensor = torch.zeros_like(x).long()
    # order = torch.arange(len(x))
    # rank_tensor[indices] = order
    return rank_tensor


class GaussianLayer(nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(10), 
            nn.Conv2d(1, 1, 21, stride=1, padding=0, bias=None)
        )

        self.weights_init()
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((21,21))
        n[10,10] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=2)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))




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

        # if self.dataset_name == 'STL10' or self.dataset_name == 'TINY_IMAGENET':
        #     img = np.transpose(img, [1, 2, 0])

        # Cutout module begins
        # xcm = int(np.random.rand()*95)
        # ycm = int(np.random.rand()*95)
        # img = self.cutout(img,xcm,ycm,24)
        #Cutout module ends

        # print(np.max(img),np.min(img))

        # img = Image.fromarray(np.uint8(img*255))

        # img = np.float32(scipy.misc.imresize(img, 2.0))
        # Optional:
        # img = img / np.max(img)

        # if self.distractor is True and self.labels[index] < 3:
        #     img = self.add_class_distractor(img,1,self.color_class[int(self.labels[index])])

        # if self.smoothing:
        #     img = gaussian_filter(img,sigma=(global_settings.global_SIGMA,global_settings.global_SIGMA,0))

        if self.transform is not None:
            img = self.transform(img)
        
        y = int(self.labels[index])

        return img, y



def rankCorrelation_variants_1D(X,Y,variant='chatterjee_normal',**kwargs):
    
    # X is 2D and Y assumed is 1D torch tensor 
    if variant == "chatterjee_normal":
        rank_Y = get_rank_argsort(Y).float()
        # rank_Y = np.digitize(Y.cpu(), np.linspace(0,1,len(X)), right=True)
        
        n = len(X)
        ind_X = torch.argsort(X)
    
        Xsorted_Y= rank_Y[ind_X]
        XsY0 = Xsorted_Y[:-1]
        XsY1 = Xsorted_Y[1:]
        diff_sum = torch.sum(torch.abs(XsY0-XsY1))
        CC = 1 - (3*diff_sum/((n**2)-1))
        return CC
    
    if variant == "chatterjee_norank":        
        min_Y = float(kwargs['min_y'])
        max_Y = float(kwargs['max_y'])
        rank_Y = (Y - min_Y)/(max_Y-min_Y)
        n = len(X)
        ind_X = torch.argsort(X)
        Xsorted_Y= rank_Y[ind_X]
        XsY0 = Xsorted_Y[:-1]
        XsY1 = Xsorted_Y[1:]
        diff_sum = torch.sum(torch.abs(XsY0-XsY1))
        CC = 1 - (3*n*diff_sum/((n**2)-1))
        return CC


print('here')


def spearmanr(pred, target, **kw):
    # pred = torchsort.soft_rank(pred, **kw)
    # target = torchsort.soft_rank(target, **kw)
    # pred = pred - pred.mean()
    # pred = pred / pred.norm()
    # target = target - target.mean()
    # target = target / target.norm()
    
    
    return stats.spearmanr(pred,target)

# from collections import Counter

def get_argsorts(X):
    Inds_X = torch.zeros_like(X)
    Inds_X = Inds_X.int()
    for i in range(X.shape[0]):
        Inds_X[i] = torch.argsort(X[i])
    return Inds_X



# def multi_regress_withgrad(X,y,p,Mul_mats,normalize=True,exponent=False):

#     # print(p)
#     X = X.unsqueeze(2)
#     N = X.shape[0]
#     if normalize == True:
#         y = y/torch.std(y,1).unsqueeze(1).repeat(1,784)
#     X = X.unsqueeze(2)
#     # y = y.unsqueeze(1)
#     X = torch.stack([torch.ones(X.shape), X],dim=2).squeeze()

#     for i in range(p-1):
#         X = torch.concat((X, (X[:,:,1]**(i+2)).unsqueeze(2)),dim=2)
#         # X = torch.concat((X, (torch.sin(alpha*X[:,:,1]*(i+1))).unsqueeze(2)),dim=2)


#     # print(X.T @ X)
#     M = torch.cat(Mul_mats,0)
#     # W = M @ y
#     W = M @ y.T
#     W = W.view(int(W.shape[0]/Mul_mats[0].shape[0]),Mul_mats[0].shape[0],y.shape[0])

#     # W = Mul_mats @ y

#     # predicted = torch.einsum('bij,bj->bi', X, W)
#     pred_final = torch.einsum('bij,bjk->bik', X, W)

#     pred_final = pred_final.permute(2,1,0)

#     # predicted = X @ W
#     err = torch.mean((y.unsqueeze(2).repeat(1,1,pred_final.shape[2])-pred_final)**2,dim=1)
#     # err = torch.mean((y-predicted)**2)

#     if exponent:
#         return torch.exp(-err)
#     else:
#         if normalize:
#             return 1-err
#         else:
#             return -err



# def get_mulmats(X,p):
#     Mul_mats = [] 
#     for i in range(X.shape[0]):
#         x = X[i] 
#         N = x.shape[0]
#         x = x.unsqueeze(1)
#         x_stack = torch.hstack([torch.ones((N, 1)), x])
#         for j in range(p-1):
#             # x_stack = torch.hstack([x_stack, x**(j+2)])
#             x_stack = torch.hstack([x_stack, torch.sin(alpha*x*(j+1))])

#         Mul_mats.append(torch.linalg.inv(x_stack.T @ x_stack) @ x_stack.T)


#     # print(X.T @ X)
#     return Mul_mats



# +
import itertools as it

class SRNLayer(nn.Module):
    def __init__(self,input_channels, output_channels, kernel_size, padding = 0, epsilon = 0.0001, num_slices=2,degree=4,exponent=True, normalize = True):
        super(SRNLayer, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,kernel_size,kernel_size,num_slices))
        torch.nn.init.xavier_normal_(self.weights,gain=0.01)
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
#     @torch.compile
    def forward(self, y2,train_status=True):
        # print(self.weights.shape)
        # if train_status:
        #     Ef = (torch.rand_like(y2)>0.1).float()
        #     y2 = Ef*y2 
        # else:
        #     print('hofida')
        
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
        #         X = torch.concat((X, (X[:,:,i]*X[:,:,j]).unsqueeze(2)),dim=2)
        
        # G = X.view(X.shape[0],1,int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1])))
        # X = nn.Unfold((3,3),padding=1)(G).permute(0,2,1)
        # # indices = torch.Tensor([2,4,6,8]).int()
        # X = X[:,:,3:7]
        
        X = torch.concat((torch.ones((X.shape[0],X.shape[1],1),requires_grad=False), X),dim=2)
        
        # M = X.pinverse()
        X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
        X_cov_inv = torch.linalg.inv(X_cov+ decay_regress)
        M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
        
        W = torch.einsum('ijk,akb->aijb',M,y)
       
        pred_final = torch.einsum('bec,abcd->abed', X, W)   
        
        
        # y3 = y.unsqueeze(1).expand(-1,pred_final.shape[1],-1,-1)
        # print(y3.shape)
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
# -

# class ShiftSRNLayer(nn.Module):
#     def __init__(self,input_channels, output_channels, kernel_size, padding = 0, epsilon = 0.0001, num_slices=2,degree=4,exponent=True, normalize = True):
#         super(ShiftSRNLayer, self).__init__()

#         self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,kernel_size,kernel_size,num_slices))
#         torch.nn.init.xavier_normal_(self.weights,gain=0.01)
#         self.exponent = exponent
#         self.kernel_size = kernel_size
#         self.normalize = normalize
#         self.input_channels = input_channels
#         self.output_channels = output_channels
#         self.num_slices = num_slices
#         self.degree = degree
#         self.shift_kernel = 3
#         self.epsilon = epsilon
#         self.padding = padding
#         self.iter =  it.combinations(np.arange(num_slices), 2)

#     def forward(self, y2,train_status=True):
#         # print(self.weights.shape)
#         # if train_status:
#         #     Ef = (torch.rand_like(y2)>0.1).float()
#         #     y2 = Ef*y2 
#         # else:
#         #     print('hofida')

#         y = nn.Unfold((self.weights.shape[2],self.weights.shape[3]),padding=self.padding)(y2)
#         # print(self.epsilon)

#         y = y + self.epsilon*torch.randn_like(y)
#         if self.normalize:
#             GG = torch.std(y,dim=1)  
#             # A =(GG<epsilon).float()*epsilon + (GG>=epsilon).float()*torch.std(y,dim=1)
#             y = y/GG.unsqueeze(1).repeat(1,y.shape[1],1)

#         X = self.weights.squeeze()
#         U = nn.Unfold((self.shift_kernel,self.shift_kernel),padding=1)(X).permute(0,2,1)
#         U = U.view(U.shape[0],U.shape[1],X.shape[1],self.shift_kernel**2).permute(0,2,1,3)
#         U = U.contiguous().view(U.shape[0],U.shape[1]*U.shape[2],U.shape[3])
#         X = U[:,:,4:6]
#         # X = X.view(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3],self.num_slices)


#         # for i in range(self.degree-1):
#         #     # X = torch.concat((X, (torch.exp(-alpha((i+2)*X[:,:,1])).unsqueeze(2)),dim=2)
#         #     X = torch.concat((X, X[:,:,0:self.num_slices]**(i+2)),dim=2)
#             # X = torch.concat((X, (X[:,:,1]**(i+2)).unsqueeze(2)),dim=2)

#         # for i,j in self.iter:
#         #     # X = torch.concat((X, (torch.exp(-alpha((i+2)*X[:,:,1])).unsqueeze(2)),dim=2)
#         #     if i!=j:
#         #         X = torch.concat((X, (X[:,:,i]*X[:,:,j]).unsqueeze(2)),dim=2)

#         # G = X.view(X.shape[0],1,int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1])))
#         # X = nn.Unfold((3,3),padding=1)(G).permute(0,2,1)
#         # # indices = torch.Tensor([2,4,6,8]).int()
#         # X = X[:,:,3:7]

#         X = torch.concat((torch.ones((X.shape[0],X.shape[1],1),requires_grad=False), X),dim=2)

#         # M = X.pinverse()
#         X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
#         X_cov_inv = torch.linalg.inv(X_cov+ decay_regress)
#         M = torch.einsum('bij,bkj->bik', X_cov_inv, X)

#         W = torch.einsum('ijk,akb->aijb',M,y)

#         pred_final = torch.einsum('bec,abcd->abed', X, W)   


#         # y3 = y.unsqueeze(1).expand(-1,pred_final.shape[1],-1,-1)
#         # print(y3.shape)
#         err = torch.mean((y.unsqueeze(1).repeat(1,pred_final.shape[1],1,1)-pred_final)**2,dim=2)

#         err = err.view(err.shape[0],err.shape[1],int(np.sqrt(err.shape[2])),int(np.sqrt(err.shape[2])))
#         # err = err/err.detach().max()
#         if self.exponent:
#             # print('here')
#             A = (torch.exp(-err)-np.exp(-1.0))/(np.exp(0)-np.exp(-1.0))
#             # return torch.exp(-err)
#             return A-0.5
#         else:
#             if self.normalize:
#                 return 1-err
#             else:
#                 return -err


# class SRNBiasLayer(nn.Module):
#     def __init__(self,input_channels, output_channels, kernel_size, padding = 0, epsilon = 0.0001, num_slices=2,degree=4,exponent=True, normalize = True):
#         super(SRNBiasLayer, self).__init__()

#         self.weights = torch.nn.Parameter(torch.zeros(output_channels, input_channels,kernel_size,kernel_size,num_slices))
#         self.weight_bias = torch.nn.Parameter(torch.zeros(output_channels, kernel_size*kernel_size*input_channels))

#         torch.nn.init.xavier_normal_(self.weights,gain=0.01)
#         self.sigma = torch.nn.Parameter(torch.ones(1,output_channels,1,1))
#         self.exponent = exponent
#         self.normalize = normalize
#         self.input_channels = input_channels
#         self.output_channels = output_channels
#         self.num_slices = num_slices
#         self.degree = degree 
#         self.epsilon = epsilon
#         self.padding = padding
#         self.iter =  it.combinations(np.arange(num_slices), 2)

#     def forward(self, y2,train_status=True):
#         # print(self.weights.shape)
#         # if train_status:
#         #     Ef = (torch.rand_like(y2)>0.1).float()
#         #     y2 = Ef*y2 
#         # else:
#         #     print('hofida')

        

        

#         y = nn.Unfold((self.weights.shape[2],self.weights.shape[3]),padding=self.padding)(y2)
#         # print(self.epsilon)

#         y = y + self.epsilon*torch.randn_like(y)

#         y = y.unsqueeze(1).repeat(1,self.weight_bias.shape[0],1,1)
#         y = y - (self.weight_bias.unsqueeze(0).unsqueeze(3).repeat(y.shape[0],1,1,y.shape[3]))

#         if self.normalize == True:
#             y = y/(torch.std(y,2)).unsqueeze(2).repeat(1,1,y.shape[2],1)

#         X = self.weights 
#         X = X.view(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3],self.num_slices)


#         for i in range(self.degree-1):
#             # X = torch.concat((X, (torch.exp(-alpha((i+2)*X[:,:,1])).unsqueeze(2)),dim=2)
#             X = torch.concat((X, X[:,:,0:self.num_slices]**(i+2)),dim=2)
#             # X = torch.concat((X, (X[:,:,1]**(i+2)).unsqueeze(2)),dim=2)

#         # for i,j in self.iter:
#         #     # X = torch.concat((X, (torch.exp(-alpha((i+2)*X[:,:,1])).unsqueeze(2)),dim=2)
#         #     if i!=j:
#         #         X = torch.concat((X, (X[:,:,i]*X[:,:,j]).unsqueeze(2)),dim=2)

#         # G = X.view(X.shape[0],1,int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1])))
#         # X = nn.Unfold((3,3),padding=1)(G).permute(0,2,1)
#         # # indices = torch.Tensor([2,4,6,8]).int()
#         # X = X[:,:,3:7]

#         # X = torch.concat((torch.ones((X.shape[0],X.shape[1],1),requires_grad=False), X),dim=2)

#         # M = X.pinverse()
#         X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
#         X_cov_inv = torch.linalg.inv(X_cov+ decay_regress)
#         M = torch.einsum('bij,bkj->bik', X_cov_inv, X)

#         W = torch.einsum('ijk,aikb->aijb',M,y)

#         pred_final = torch.einsum('bec,abcd->abed', X, W)   


#         # y3 = y.unsqueeze(1).expand(-1,pred_final.shape[1],-1,-1)
#         # print(y3.shape)
#         err = torch.mean((y-pred_final)**2,dim=2)

#         err = err.view(err.shape[0],err.shape[1],int(np.sqrt(err.shape[2])),int(np.sqrt(err.shape[2])))
#         # err = err/err.detach().max()
#         if self.exponent:
#             # print('here')
#             # A = (torch.exp(-err)-np.exp(-1.0))/(np.exp(0)-np.exp(-1.0))
#             # return A-0.5
#             return torch.exp(-err/(self.sigma.repeat(err.shape[0],1,err.shape[2],err.shape[3])**2+0.0001))
#         else:
#             if self.normalize:
#                 return 1-err
#             else:
#                 return -err


# def SRModule_Conv(X,y2,p, padding,decay_regress, epsilon=0.0001, normalize=True,exponent=False):

#     # print(p)
#     # Mul_mats = [] 
#     # for i in range(X.shape[0]):
#     #     x = X[i] 
#     #     N = x.shape[0]
#     #     x = x.unsqueeze(1)
#     #     x_stack = torch.hstack([torch.ones((N, 1)), x])
#     #     for j in range(p-1):
#     #         x_stack = torch.hstack([x_stack, x**(j+2)])
#     #         # x_stack = torch.hstack([x_stack, torch.sin(alpha*x*(j+1))])

#     #     # Mul_mats.append(torch.linalg.inv(x_stack.T @ x_stack) @ x_stack.T)
#     #     D,info = torch.linalg.inv_ex(x_stack.T @ x_stack)
#     #     Mul_mats.append(D @ x_stack.T)

#     # print(p)
#     # print('mm')
#     # X = X.unsqueeze(2)
#     # y = y.repeat(1,10,1,1)
#     # print(y2.shape)
#     # if normalize == True:
#     #     D = y2.view(y2.shape[0],y2.shape[1],y2.shape[2]*y2.shape[3])
#     #     y2 = y2/torch.std(D,dim=2).unsqueeze(2).unsqueeze(3).repeat(1,1,y2.shape[2],y2.shape[3])
#     # epsilon = 0.001


#     # print(padding)
#     # print(y2.shape)


#     y = nn.Unfold((X.shape[2],X.shape[3]),padding=padding)(y2)
#     y = y + epsilon*torch.randn_like(y)
#     if normalize:
#         GG = torch.std(y,dim=1)  
#         # A =(GG<epsilon).float()*epsilon + (GG>=epsilon).float()*torch.std(y,dim=1)
#         y = y/GG.unsqueeze(1).repeat(1,y.shape[1],1)
#     # if normalize:
#     #     y = y/(torch.std(y,dim=1)+epsilon).unsqueeze(1).repeat(1,y.shape[1],1)
#     N = X.shape[0]
#     if len(y.shape)==1:
#         y = y.unsqueeze(0)


#     # if normalize == True:
#     #     y = y/torch.std(y,1).unsqueeze(1).repeat(1,X.shape[1])
#         # y2,ind = torch.max(y,1)
#         # y = y/y2.unsqueeze(1).repeat(1,X.shape[1])
#     X = X.view(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3])
#     X = X.unsqueeze(2)
#     # y = y.unsqueeze(1)
#     X = torch.stack([torch.ones(X.shape), X],dim=3).squeeze()

#     for i in range(p-1):
#         # X = torch.concat((X, (torch.exp(-alpha((i+2)*X[:,:,1])).unsqueeze(2)),dim=2)
#         X = torch.concat((X, (X[:,:,1]**(i+2)).unsqueeze(2)),dim=2)
#         # X = torch.concat((X, (torch.sin(alpha*X[:,:,1]*(i+1))).unsqueeze(2)),dim=2)

#     X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
#     # X_cov = torch.einsum('abdc,abce->abde', X.permute(0,1,3,2),X)

#     # print(decay_regress)
#     # X_cov_inv = torch.linalg.inv(X_cov+ decay_regress)
#     X_cov_inv = torch.linalg.inv(X_cov+ decay_regress)
#     # print(torch.mean(torch.abs(X_cov_inv)))
#     # print(torch.mean(torch.abs(X)))
#     M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
#     # M = torch.einsum('abcd,abed->abce', X_cov_inv, X)
#     # W_ = torch.einsum('ijk,bik->ijb',M,y_)
#     W = torch.einsum('ijk,akb->aijb',M,y)


#     # W = Mul_mats @ y

#     # predicted = torch.einsum('bij,bj->bi', X, W)
#     # pred_final = torch.einsum('bij,bjk->bik', X, W)
#     pred_final = torch.einsum('bec,abcd->abed', X, W)
#     # y = y.unsqueeze(1).repeat(1,pred_final.shape[1],1,1)
#     # pred_final = pred_final.permute(2,1,0)

#     # hada_mult = hada_mult**2
#     # hada_mult = hada_mult/ (torch.sum(hada_mult,1).unsqueeze(1).repeat(1,pred_final.shape[1]) + 0.000001)
#     # hada_mult = hada_mult.permute(1,0).unsqueeze(0).repeat(pred_final.shape[0],1,1)
#     # predicted = X @ W
#     # weights = X[:,:,1].permute(1,0).unsqueeze(0).repeat(pred_final.shape[0],1,1)
#     # err_mat = (y.unsqueeze(2).repeat(1,1,pred_final.shape[2])-pred_final)**2
#     # err = torch.sum(err_mat*weights,dim=1)/torch.sum(weights,dim=1)
#     # X_prime = torch.abs(X[:,:,1].permute(1,0).unsqueeze(0).repeat(400,1,1))

    

#     err = torch.mean((y.unsqueeze(1).repeat(1,pred_final.shape[1],1,1)-pred_final)**2,dim=2)
#     # err = torch.mean((y.unsqueeze(1).unsqueeze_(1)-pred_final)**2,dim=2)


#     # eerrrrrr
#     # err = torch.sum((y.unsqueeze(2).repeat(1,1,pred_final.shape[2])-pred_final)**2*hada_mult,dim=1)


#     # alopha = 0.996
#     # err = err*((err<alopha).float()) +alopha*((err>=alopha).float())

#     # M = (y.unsqueeze(2).repeat(1,1,pred_final.shape[2])-pred_final)**2

#     # # with torch.no_grad():
#     # E = torch.exp(-M)
#     # E = E/torch.sum(E,2).unsqueeze(2).repeat(1,1,M.shape[2])

#     # err = torch.mean(E*M,dim=1)


#     # err = torch.mean((y-predicted)**2)
#     # print(torch.min(err))
#     err = err.view(err.shape[0],err.shape[1],int(np.sqrt(err.shape[2])),int(np.sqrt(err.shape[2])))
#     # err = err/err.detach().max()
#     if exponent:
#         # print('here')
#         A = (torch.exp(-err)-np.exp(-1.0))/(np.exp(0)-np.exp(-1.0))
#         # return torch.exp(-err)
#         return A-0.5
#     else:
#         if normalize:
#             return 1-err
#         else:
#             return -err

    

    

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



# +
# from torcheval.metrics import R2Score
# -

def rankCorrelation_variants_multiD(X,Y,Inds_X,Mul_mats,variant='chatterjee_normal',**kwargs):
    
    # print('evoked')
    # X is 2D and Y assumed is 1D torch tensor 
    if variant == "chatterjee_normal":
        rank_Y = get_rank_argsort(Y).float()
        # rank_Y = torch.from_numpy(np.digitize(Y.cpu(), np.linspace(0,1,len(Y)), right=True)).cuda()

        # CC_array = torch.zeros(X.shape[0])
        n = len(X[0])
        Xsorted_Y = rank_Y[Inds_X]
        XsY0 = Xsorted_Y[:,:-1]
        XsY1 = Xsorted_Y[:,1:]
        diff_sum = torch.sum(torch.abs(XsY0-XsY1),1)
        CC_array = 1 - (3*diff_sum/((n**2)-1))
        
        # for i in range(X.shape[0]):
        #     n = len(X[i])
        #     ind_X = Inds_X[i]
        
        #     Xsorted_Y= rank_Y[ind_X]
        #     XsY0 = Xsorted_Y[:-1]
        #     XsY1 = Xsorted_Y[1:]
        #     diff_sum = torch.sum(torch.abs(XsY0-XsY1))
        #     CC = 1 - (3*diff_sum/((n**2)-1))
        #     CC_array[i] = CC
        
        return CC_array 
    if variant == "chatterjee_squared":
        rank_Y = get_rank_argsort(Y).float()
        # rank_Y = torch.from_numpy(np.digitize(Y.cpu(), np.linspace(0,1,len(Y)), right=True)).cuda()

        # CC_array = torch.zeros(X.shape[0])
        n = len(X[0])
        Xsorted_Y = rank_Y[Inds_X]
        XsY0 = Xsorted_Y[:,:-1]
        XsY1 = Xsorted_Y[:,1:]
        diff_sum = torch.mean((XsY0-XsY1)**2,1)/(n**2)
        CC_array = diff_sum
        # CC_array = 1 - (3*diff_sum/((n**2)-1))
        
        # for i in range(X.shape[0]):
        #     n = len(X[i])
        #     ind_X = Inds_X[i]
        
        #     Xsorted_Y= rank_Y[ind_X]
        #     XsY0 = Xsorted_Y[:-1]
        #     XsY1 = Xsorted_Y[1:]
        #     diff_sum = torch.sum(torch.abs(XsY0-XsY1))
        #     CC = 1 - (3*diff_sum/((n**2)-1))
        #     CC_array[i] = CC
        
        return CC_array 
    
    if variant == "chatterjee_normal_reverse":
        ind_Y = torch.argsort(Y)
        
        CC_array = torch.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            n = len(X[i])
            rank_X = get_rank_argsort(X[i]).float()
            
            Xsorted_Y= rank_X[ind_Y]
            XsY0 = Xsorted_Y[:-1]
            XsY1 = Xsorted_Y[1:]
            diff_sum = torch.sum(torch.abs(XsY0-XsY1))
            CC = 1 - (3*diff_sum/((n**2)-1))
            CC_array[i] = CC
        
        return CC_array 
        
    if variant == "chatterjee_norank":        
        # min_Y = float(kwargs['min_y'])
        # max_Y = float(kwargs['max_y'])
        min_Y = float(torch.min(Y))
        max_Y = float(torch.max(Y))
        
        rank_Y = (Y - min_Y)/(max_Y-min_Y)
        n = len(X[0])
        
        Xsorted_Y= rank_Y[Inds_X]
        XsY0 = Xsorted_Y[:,:-1]
        XsY1 = Xsorted_Y[:,1:]
        # diff_sum = torch.sum(torch.abs(XsY0-XsY1),1)
        # CC_array = 1 - (3*n*diff_sum/((n**2)-1))
        diff_mean = torch.mean(torch.abs(XsY0-XsY1),1)
        CC_array = diff_mean
        return CC_array
    
        # CC_array[i] = CC
        # CC_array = torch.zeros(X.shape[0])

        # for i in range(X.shape[0]):
        #     n = len(X[i])
        #     ind_X = Inds_X[i].long()
        #     Xsorted_Y= rank_Y[ind_X]
        #     XsY0 = Xsorted_Y[:-1]
        #     XsY1 = Xsorted_Y[1:]
        #     diff_sum = torch.sum(torch.abs(XsY0-XsY1))
        #     CC = 1 - (3*n*diff_sum/((n**2)-1))
        #     CC_array[i] = CC
        # return CC_array
    
    if variant == "normalized_smoothness":
        
        CC_array = torch.zeros(X.shape[0])
        n = len(X[0])
        Xsorted_Y= Y[Inds_X]
        XsY0 = Xsorted_Y[:,:-1]
        XsY1 = Xsorted_Y[:,1:]
        diff_sum = torch.mean(torch.abs(XsY0-XsY1)**2,1)
        CC_array = 1 - (diff_sum/torch.var(Y))
        return CC_array
    
    if variant == "normalized_smoothness_monotonic":
        
        CC_array = torch.zeros(X.shape[0])
        n = len(X[0])
        Xsorted_Y= Y[Inds_X]
        XsY0 = Xsorted_Y[:,:-1]
        XsY1 = Xsorted_Y[:,1:]
        diff_sum = torch.mean(torch.abs(XsY0-XsY1)**2*((XsY0-XsY1)>=0).float(),1)
        CC_array = 1 - (diff_sum/torch.var(Y))
        return CC_array
    
    if variant == "normalized_absolute_smoothness":
        
        CC_array = torch.zeros(X.shape[0])
        n = len(X[0])
        Xsorted_Y= Y[Inds_X]
        XsY0 = Xsorted_Y[:,:-1]
        XsY1 = Xsorted_Y[:,1:]
        diff_sum = torch.mean(torch.abs(XsY0-XsY1),1)
        CC_array = 1 - (diff_sum/torch.std(Y))
        return CC_array
    
    if variant == "regress_input_linear":
        CC_array = torch.zeros(X.shape[0])
        
        # metric = R2Score(device='cuda')
        # T = time.time()
        for i in range(X.shape[0]):
            CC_array[i] = regress_withgrad(X[i],Y,p=DEGREE,normalize = True,exponent = True,Mul_mat=Mul_mats[i])
        # print(400*(time.time()-T))
        return CC_array
    
    # if variant == "regress_batch":
        # layer_out = multi_regress_withgrad(X,Y,p=DEGREE,Mul_mats,normalize=True,exponent=False)
        # return layer_out
    
    if variant == "chatterjee_monotonic_squared":
        rank_Y = get_rank_argsort(Y).float()
        # rank_Y = torch.from_numpy(np.digitize(Y.cpu(), np.linspace(0,1,len(Y)), right=True)).cuda()

        # CC_array = torch.zeros(X.shape[0])
        n = len(X[0])
        Xsorted_Y = rank_Y[Inds_X]
        XsY0 = Xsorted_Y[:,:-1]
        XsY1 = Xsorted_Y[:,1:]
        # diff_sum = torch.sum(torch.abs(XsY0-XsY1)*((XsY0-XsY1)>0).float(),1)
        # CC_array = 1 - (3*diff_sum/((n**2)-1))
        CC_array = 1- (torch.mean(torch.abs(XsY0-XsY1)**2*((XsY0-XsY1)>=0).float(),1)/n**3)
        return CC_array
        
    if variant == "chatterjee_monotonic":
        rank_Y = get_rank_argsort(Y).float()
        # rank_Y = torch.from_numpy(np.digitize(Y.cpu(), np.linspace(0,1,len(Y)), right=True)).cuda()

        # CC_array = torch.zeros(X.shape[0])
        n = len(X[0])
        Xsorted_Y = rank_Y[Inds_X]
        XsY0 = Xsorted_Y[:,:-1]
        XsY1 = Xsorted_Y[:,1:]
        # diff_sum = torch.sum(torch.abs(XsY0-XsY1)*((XsY0-XsY1)>0).float(),1)
        # CC_array = 1 - (3*diff_sum/((n**2)-1))
        CC_array = 1- (torch.mean(torch.abs(XsY0-XsY1)*((XsY0-XsY1)>=0).float(),1)/n)
    
        
        # for i in range(X.shape[0]):
        #     n = len(X[i])
        #     ind_X = Inds_X[i]
        
        #     Xsorted_Y= rank_Y[ind_X]
        #     XsY0 = Xsorted_Y[:-1]
        #     XsY1 = Xsorted_Y[1:]
        #     diff_sum = torch.sum(torch.abs(XsY0-XsY1))
        #     CC = 1 - (3*diff_sum/((n**2)-1))
        #     CC_array[i] = CC
        
        return CC_array 
    
    

    
    
    
        # for i in range(X.shape[0]):
        #     n = len(X[i])
        #     ind_X = torch.argsort(X[i])
        #     Xsorted_Y= Y[ind_X]
        #     XsY0 = Xsorted_Y[:-1]
        #     XsY1 = Xsorted_Y[1:]
        #     diff_sum = torch.mean(torch.abs(XsY0-XsY1)**2)
        #     CC = 1 - (diff_sum/torch.var(Y))
        #     CC_array[i] = CC
        # return CC_array
    
    
    if variant == "spearman":
        
        CC_array = torch.zeros(X.shape[0])

        for i in range(X.shape[0]):
            CC_array[i] = stats.spearmanr((Y.cpu().numpy()), X[i].cpu().numpy()).statistic
            
        return CC_array
    # if variant == "chatterjee_"

        

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



# class net_primal_mnist(nn.Module):

#     def __init__(self):
#         super(net_primal_mnist, self).__init__()

#         self.conv1 = nn.Conv2d(1,64,3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
#         self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

#         # self.conv8 = nn.Conv2d(64, 64, 3)
#         # self.conv9 = nn.Conv2d(64, 64, 3)
#         # self.conv10 = nn.Conv2d(64, 64, 3)


#         self.mpool1 = nn.AvgPool2d(2)
#         self.mpool2 = nn.AvgPool2d(2)
#         self.mpool3 = nn.AvgPool2d(2,padding=1)
#         self.mpool4 = nn.AvgPool2d(4)
#         # self.mpool5 = nn.MaxPool2d(2)


#         self.bnorm1 = nn.BatchNorm2d(64)
#         self.bnorm2 = nn.BatchNorm2d(128)
#         self.bnorm3 = nn.BatchNorm2d(128)
#         self.bnorm4 = nn.BatchNorm2d(128)


#         self.bnorm_fc = nn.BatchNorm1d(128)
#         self.fc1 = nn.Linear(128,128)
#         self.fc2 = nn.Linear(128,10,1)

#         self.feat_net = nn.Sequential(

#             # self.conv1,
#             self.bnorm1,
#             nn.ReLU(inplace=True),
#             self.mpool1,

#             self.conv2,
#             self.bnorm2,
#             nn.ReLU(inplace=True),
#             self.mpool2,
#             #
#             self.conv3,
#             self.bnorm3,
#             nn.ReLU(inplace=True),
#             self.mpool3,
#             # #
#             self.conv4,
#             self.bnorm4,
#             nn.ReLU(inplace=True),
#             self.mpool4
#             #

#         )

#         self.postbag_network = nn.Sequential(

#             # self.conv3,
#             # self.bnorm3,
#             # nn.ReLU(inplace=True),
#             # self.mpool3,


#             # self.conv5,
#             # self.bnorm5,
#             # nn.ReLU(inplace=True),
#             # self.mpool5,

#         )

#         self.bnorm_fc = nn.BatchNorm1d(128)
#         self.fc1 = nn.Linear(128,128)
#         self.fc2 = nn.Linear(128,10,1)


#     def forward(self, x):
#         x_prebag = self.prebag_network(x)
#         x_postbag = self.postbag_network(x_prebag)

#         x = x_postbag.view(x_postbag.size(0), -1)
#         x = F.relu(self.bnorm_fc(self.fc1(x)))
#         x = self.fc2(x)
#         return x

  

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
        # print(kernels[0])
        self.srn1 = SRNLayer(input_channels, 64, kernels[0],padding=self.padding[0],epsilon = epsilons[0],num_slices=8,degree=1)
        
        # ----------------------- Old code----------------------------
        # if self.mode =='regress_batch':
        #     self.conv1 = nn.Conv2d(input_channels, layers[0], kernels[0])
        #     self.pool1 = nn.AvgPool2d(kernel_size=(2,2))
        #     self.pool2 = nn.AvgPool2d(kernel_size=(2,2))
            
        #     self.conv2 = nn.Conv2d( layers[0], layers[1], kernels[1]) 
            
            
            
        #     self.bn1 = nn.BatchNorm2d(layers[0])
        #     self.bn2 = nn.BatchNorm2d(layers[0],affine=False)
        #     self.bn3 = nn.BatchNorm2d(1250)
        #     self.fc_direct = nn.Conv2d(1250, classes, 1)

        # ----------------------- Old code----------------------------
        
        if self.mode =='normal':
            self.conv1 = nn.Conv2d(3,64,3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
            self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
            
            self.mpool1 = nn.AvgPool2d(2)
            self.mpool2 = nn.MaxPool2d(2)
            self.mpool3 = nn.MaxPool2d(2)
            self.mpool4 = nn.MaxPool2d(8)
            
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(128)
            
            
            self.bnorm_fc = nn.BatchNorm2d(512)
            self.fc1 = nn.Conv2d(512,512,1)
            self.fc2 = nn.Conv2d(512,10,1)
            
            self.feat_net = nn.Sequential(
    
                # self.conv1,
#                 self.mpool1,
#                 self.bn1,
                # nn.ReLU(inplace=True),
                
                self.conv1,
                self.bn1,
                nn.ReLU(inplace=True),
                # self.mpool1,
    
                self.conv2,
                self.bn2,
                nn.ReLU(inplace=True),
                self.mpool2,
                #
                self.conv3,
                self.bn3,
                nn.ReLU(inplace=True),
#                 self.mpool3,
                # #
                self.conv4,
                self.bn4,
                nn.ReLU(inplace=True),
                self.mpool4
                #
            )

        self.drop = nn.Dropout(p=self.dropping)
    
            
        self.relu = nn.ReLU(inplace=True)
        # self.fc_direct0 = nn.Conv2d(layers[0], 200, 1)
        # self.bn0 = nn.BatchNorm2d(200)
        # self.fc_direct = nn.Conv2d(1250, classes, 1)
    
            
    # def regress_batch_layer(self,weights,inputs,Mul_mats):
    #     # with torch.no_grad():
    #     layer_out = multi_regress_withgrad(weights,inputs,DEGREE,Mul_mats,normalize=True,exponent=False)
    #     return layer_out.unsqueeze(2).unsqueeze(3)
    
    # def conv_regress_batch_standalone(self,weights,inputs,p,epsilon, padding = 0, normalize=True, exponent = True):
    #     # with torch.no_grad():
    #     # print(weights[0][0])
    #     # print(padding)
    #     layer_out = SRModule_Conv(weights,inputs,p,padding, self.decay_regress,epsilon,  normalize=normalize,exponent=exponent)
    #     return layer_out
        
    
        
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
            # x = self.conv_regress_batch_standalone(self.conv1.weight,
                                            # x,epsilon = self.epsilons[0],padding = self.padding[0], p=DEGREE[0],exponent=True)
#             x = self.srn1(x,net.training)
            x = self.feat_net(x)
            x_errs = x
                    
                
        # elif self.mode =='just_normal':
            
        #     # for i in range(len(self.convs)):
        #         # x = self.convs[i](x)
        #         # # x = self.drop(x)
        #         # x = self.relu(x)
        #     x = self.conv1(x)
        #     x = self.relu(x)
        
        #         # print(torch.mean(x))
        # elif self.mode == 'regress_batch':
        #     # for i in range(len(self.convs)):
        #         # T = time.time()
        #         # x = self.regress_batch_layer(self.convs[i].weight.squeeze(),
        #                                      # x.squeeze(),self.Mul_mats[i])
                                             
        #     # print('baby step inside')
        #     # print(x.shape)
        #     x = self.conv_regress_batch_standalone(self.conv1.weight,
        #                                  x,epsilon = self.epsilons[0],padding = self.padding[0], p=DEGREE[0],exponent=True)
        #     # print(x.shape)
        #     # print(self.padding[0])
        #     # print(x_errs.shape)
        #     x = self.pool1(x)
        #     # print(x.shape)
        #     # if self.use_bn:
        #     # x = self.relu(self.bn2(x))
        #     # print(x.shape)
        #     x_errs = self.conv_regress_batch_standalone(self.conv2.weight,
        #                                  x,epsilon = self.epsilons[1],padding = self.padding[1], p=DEGREE[1],exponent=True)
        #     # print(x.shape)
        #     x = self.pool2(x_errs)
            
        #     # x = self.relu(self.bn3(x))
        #     # print(x.shape)
            
        #     # print(x.shape)
        #         # x = self.bias1(x)
        #     # else:
        #     #     x = x_errs
        #     x = self.drop(x)
            
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        # xm = self.bn3(xm)
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
        # xm = self.fc_direct(xm)
        # print(xm.shape)
        xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)
        
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
            allouts = net(inputs)

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
            allouts = net(inputs)

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


# +
def train_network_normal(net,trainloader,testloader,  init_rate,epochs,weight_decay):
#     net = torch.compile(net,mode='max-autotune')
    net = net.cuda()
    net = net.train()
#     optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=1e-6,nesterov=True)
    optimizer = optim.Adam(net.parameters(), lr=init_rate, weight_decay=0)
    # decayRate = 0.9
    # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
#     scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma_learning)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], last_epoch=-1)

    criterion = nn.CrossEntropyLoss()
    
    init_epoch = 0
    all_train_losses = []
    bazinga = 0 
    train_loss_min = 9999
    for epoch in range(epochs):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # scheduler.step()
        print('epoch: ' + str(epoch))
        train_loss = []
        loss_weights = [] 
        # if epoch>epochs/2:
        #     net.decay_errors = 0
        #     bazinga = 1
        # T = time.time()'' 
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
#             print(inputs_left.shape)
#             print(inputs_right.shape)
#             inputs = torch.cat((inputs_left,inputs_right),dim=1)
#             print(inputs.shape)
            inputs = inputs.cuda()
            labels = labels.cuda()
#             print(info)
            # inputs = inputs.cuda()
            # labels = labels.cuda()
#             print(inputs.max())
#             print(inputs.min())
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # print(3)
            # print(inputs.dtype)
            allouts = net(inputs)
            # print(2)
            # l2_penalty = weight_decay * sum([(p**2).sum() for p in net.fc_direct.parameters()])
            # l1_penalty = weight_decay * sum([p.abs().sum() for p in net.fc_direct.parameters()])
            # print(net.decay_errors)
            loss = criterion(allouts, labels.long()) #+ net.decay_errors*torch.mean(-x_errs)
            loss.backward()
            train_loss.append(loss.item())
            loss_weights.append(len(labels))
            
            # print(1)
            # print(torch.mean(1-x_errs))
            optimizer.step()
        
#         for g in optimizer.param_groups:
#             g['lr'] = init_rate* (0.5 ** (epoch // 50))

        # if np.mod(epoch+1,10)==0:
        #     accuracy = test_network(deepcopy(net), testloader, test_labels,0)
        #     print('Accuracy Watch, Epoch ' + str(epoch+1) +":"+str(accuracy))
        
        # if np.mod(epoch,10) == 0:
        #     my_lr_scheduler.step()
        # print("Time:",time.time()-T)
            # print(0)
            # print('here')
        
        all_train_losses.append(np.average(np.array(train_loss),weights=np.array(loss_weights)))
        print(all_train_losses[-1])
        if all_train_losses[-1] < train_loss_min:
            train_loss_min = copy(all_train_losses[-1])
            net_best = deepcopy(net)
            print('humbo')
        # print('break')
        
    # train_loss = train_loss/len(trainloader.sampler)
#     train_loss = np.mean(all_train_losses[-5:])
    # train_loss = all_train_losses[-1]
    net_best = net_best.eval()
    return net_best,all_train_losses


# +
def train_network_stereo(net,trainloader,testloader,init_rate,epochs,weight_decay):
#     net = torch.compile(net,mode='max-autotune')
    net = net.cuda()
    net = net.train()
#     optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=1e-6,nesterov=True)
    optimizer = optim.Adam(net.parameters(), lr=init_rate, weight_decay=0)
    # decayRate = 0.9
    # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
#     scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma_learning)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], last_epoch=-1)

    criterion = nn.CrossEntropyLoss()
    
    init_epoch = 0
    all_train_losses = []
    bazinga = 0 
    train_loss_min = 9999
    for epoch in range(epochs):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # scheduler.step()
        print('epoch: ' + str(epoch))
        train_loss = []
        loss_weights = [] 
        # if epoch>epochs/2:
        #     net.decay_errors = 0
        #     bazinga = 1
        # T = time.time()'' 
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs_left,inputs_right, labels,info = data
#             print(inputs_left.shape)
#             print(inputs_right.shape)
            inputs = torch.cat((inputs_left,inputs_right),dim=1)
#             print(inputs.shape)
            inputs = inputs.cuda()
            labels = labels.cuda()
#             print(inputs.shape)
#             print(info)
            # inputs = inputs.cuda()
            # labels = labels.cuda()
#             print(inputs.max())
#             print(inputs.min())
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # print(3)
            # print(inputs.dtype)
            allouts = net(inputs)
            # print(2)
            # l2_penalty = weight_decay * sum([(p**2).sum() for p in net.fc_direct.parameters()])
            # l1_penalty = weight_decay * sum([p.abs().sum() for p in net.fc_direct.parameters()])
            # print(net.decay_errors)
            loss = criterion(allouts, labels.long()) #+ net.decay_errors*torch.mean(-x_errs)
            loss.backward()
            train_loss.append(loss.item())
            loss_weights.append(len(labels))
            
            # print(1)
            # print(torch.mean(1-x_errs))
            optimizer.step()
        
#         for g in optimizer.param_groups:
#             g['lr'] = init_rate* (0.5 ** (epoch / 100))
# #             print(g['lr'])

        # if np.mod(epoch+1,10)==0:
        #     accuracy = test_network(deepcopy(net), testloader, test_labels,0)
        #     print('Accuracy Watch, Epoch ' + str(epoch+1) +":"+str(accuracy))
        
        # if np.mod(epoch,10) == 0:
        #     my_lr_scheduler.step()
        # print("Time:",time.time()-T)
            # print(0)
            # print('here')
        
        all_train_losses.append(np.average(np.array(train_loss),weights=np.array(loss_weights)))
        print(all_train_losses[-1])
        
        if all_train_losses[-1] < train_loss_min:
            train_loss_min = copy(all_train_losses[-1])
            net_best = deepcopy(net)
            print('humbo')
        # print('break')
        
    # train_loss = train_loss/len(trainloader.sampler)
#     train_loss = np.mean(all_train_losses[-5:])
    # train_loss = all_train_losses[-1]
    net_best = net_best.eval()
    return net_best,all_train_losses


# -

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
    elif dataset_name == 'NORB':
        train_loader, valid_loader, test_loader = load_small_norb(batch_size)
        print(train_loader.dataset.data.shape)
        test_loader_norb = load_norb(batch_size)
        return train_loader, test_loader,test_loader_norb

    dataset.data = dataset.data[:training_size]
    dataset.targets = dataset.targets[:training_size]

#     transforms_list = transforms.Compose(
#                    [ transforms.ToPILImage(),
#                     transforms.ToDevice("cuda"),
# #                     transforms.RandomRotation(30),
#                     transforms.RandomAffine(degrees=20, translate=(0.2,0.2), scale=(0.8, 1.2),shear = 20),
#                    transforms.ToTensor()]

#     )


    # print(dataset.data.flatten(1,len(dataset.data.shape)-1).unsqueeze(2).unsqueeze(3).shape)
    # dataset.data = dataset.data.flatten(1,len(dataset.data.shape)-1).unsqueeze(2).unsqueeze(3)
    
    dataset.data = dataset.data.cuda()  #train_dataset.train_data is a Tensor(input data)
    dataset.targets = dataset.targets.cuda()
    
    if labelnoise>0:
        print('H')
        label_max = torch.max(dataset.targets).cpu().numpy()
        temp_targets = dataset.targets.clone() 
        # noise_labels = copy(dataset.targets[torch.randperm(torch.numel(dataset.targets))])
        rand_indices = torch.randperm(torch.numel(dataset.targets))
        num_to_change = int(labelnoise*float(torch.numel(dataset.targets)))
        to_change = rand_indices[0:num_to_change]
        
        for i in range(len(to_change)):
            dataset.targets[to_change[i]] = rand_another(dataset.targets[to_change[i]], label_max)
        
    
    # dataset_test.data = dataset_test.data.flatten(1,len(dataset_test.data.shape)-1).unsqueeze(2).unsqueeze(3)

    dataset_test.data = dataset_test.data.cuda()  #train_dataset.train_data is a Tensor(input data)
    dataset_test.targets = dataset_test.targets.cuda()
    
    if rank_convert:
        print('R')
        dataset.data = rank_convert_data(dataset.data)
        dataset_test.data = rank_convert_data(dataset_test.data)
    
    # if atanh_convert>0: 
    #     print('A')
    #     dataset.data = map_inf(dataset.data,atanh_convert)
    #     dataset_test.data = map_inf(dataset_test.data,atanh_convert)
        
    #     print(torch.max(dataset.data))
    #     print(torch.min(dataset.data))
        
        # dataset.data =  dataset.data/torch.sqrt(1-(0.9*dataset.data**2))
        # dataset_test.data =  dataset_test.data/torch.sqrt(1-(0.9*dataset_test.data**2))
        
        # print('ere')
    
    # dataset.data,mean_vector,std_vector = normalize_data(dataset.data)
    # dataset_test.data = normalize_data(dataset_test.data,mean_vector=mean_vector,std_vector=std_vector)
    
    my_dataset = Dataset(dataset_name, dataset.data, dataset.targets)
    my_dataset_test = Dataset(dataset_name, dataset_test.data, dataset_test.targets)

    trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                          shuffle=True,generator=torch.Generator(device='cuda'), num_workers=0)
    testloader = torch.utils.data.DataLoader(my_dataset_test, batch_size=batch_size,
                                          shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)
    
    testloader_adversary = torch.utils.data.DataLoader(my_dataset_test, batch_size=1,
                                          shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)

    
    return dataset,dataset_test, my_dataset,my_dataset_test,trainloader,testloader,testloader_adversary


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


def rank_convert_data(data,**kwargs):
    
    for i in range(len(data)):
        temp,data[i] = torch.unique(data[i],return_inverse=True)
        # dataset.data = dataset.data
    
    return data


def  create_crank_generators(net, dataset,dataset_test, feat_shape,**kwargs):
    
    net = net.eval()
    net.update_inds()
    # net.update_mul_mats()
    
    
    new_data = torch.zeros((dataset.data.shape[0],feat_shape[1],1,1))
    new_test_data= torch.zeros((dataset_test.data.shape[0],feat_shape[1],1,1))
    
    with torch.no_grad():
        for i in np.arange(0,len(dataset.data),kwargs['batch_size']):
            allouts,feats = net(dataset.data[i:np.minimum(i+kwargs['batch_size'],len(new_data))])
            print(torch.mean(feats))
            new_data[i:np.minimum(i+kwargs['batch_size'],len(new_data))] = feats.detach()
        
        new_data,mean_vector,std_vector = normalize_data(new_data)
        # torch.mean(new_data,0).repeat(1000,1,1,1)
    
        for i in np.arange(0,len(dataset_test.data),kwargs['batch_size']):
            allouts,feats = net(dataset_test.data[i:np.minimum(i+kwargs['batch_size'],len(new_test_data))])
            new_test_data[i:np.minimum(i+kwargs['batch_size'],len(new_test_data))] = feats.detach()
        
        new_test_data = normalize_data(new_test_data,mean_vector=mean_vector,std_vector=std_vector)

    my_dataset = Dataset(dataset_name, new_data, dataset.targets)
    my_dataset_test = Dataset(dataset_name, new_test_data, dataset_test.targets)
    
    crank_trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                          shuffle=True,generator=torch.Generator(device='cuda'), num_workers=0)
    crank_testloader = torch.utils.data.DataLoader(my_dataset_test, batch_size=batch_size,
                                          shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)
    
    return my_dataset,my_dataset_test,crank_trainloader,crank_testloader

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


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
            all_outs = net(inputs)
            predicted = torch.argmax(all_outs,1)

            correct = correct + torch.sum(predicted == labels)
    accuracy = float(correct) / float(len(test_labels))
    return accuracy

def test_network(net, testloader,input_noise=0):
    net = net.eval()

    correct = torch.tensor(0)
    dataiter = iter(testloader)
    total = torch.tensor(0)
    # total = 0 
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
#             print(inputs_left.shape)
#             print(inputs_right.shape)
#             inputs = torch.cat((inputs_left,inputs_right),dim=1)
#             print(inputs.shape)
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs = inputs  
            all_outs = net(inputs)
            predicted = torch.argmax(all_outs,1)
            correct = correct + torch.sum(predicted == labels)
            total = total+ len(labels)
    accuracy = float(correct) / float(total)
    return accuracy

def test_network_pairs(net, testloader,input_noise=0):
    net = net.eval()

    correct = torch.tensor(0)
    dataiter = iter(testloader)
    total = torch.tensor(0)
    # total = 0 
    total_time = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs_left,inputs_right, labels,info = data
            inputs = torch.cat((inputs_left,inputs_right),dim=1)
#             print(inputs_left.shape)
#             print(inputs_right.shape)
#             inputs = torch.cat((inputs_left,inputs_right),dim=1)
#             print(inputs.shape)
            inputs = inputs.cuda() 
        
            labels = labels.cuda()
            TT = time.time()
            all_outs = net(inputs)
            total_time += time.time() - TT 
            predicted = torch.argmax(all_outs,1)
            correct = correct + torch.sum(predicted == labels)
            total = total+ len(labels)
    accuracy = float(correct) / float(total)
    print(total_time/float(total))
    return accuracy


def test_network_pairs_norb(net, testloader,input_noise=0):

    net = net.eval()

    correct = torch.tensor(0)
    dataiter = iter(testloader)
    total = torch.tensor(0)
    # total = 0 
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs_left,inputs_right, labels,info = data
            inputs = torch.cat((inputs_left,inputs_right),dim=1)
#             print(inputs_left.shape)
#             print(inputs_right.shape)
#             inputs = torch.cat((inputs_left,inputs_right),dim=1)
#             print(inputs.shape)
            
            inputs = inputs.cuda() 
            indices = torch.argwhere(labels!=5)
            labels = labels.cuda()
            all_outs = net(inputs)
            predicted = torch.argmax(all_outs,1)
            correct = correct + torch.sum(predicted[indices] == labels[indices])
            total = total+ len(indices)
            
    accuracy = float(correct) / float(total)
    print(total)
    return accuracy


# +
import kornia 

def test_network_pairs_norb_tta(net, testloader,input_noise=0):
    RC = kornia.augmentation.RandomCrop((64,64))
    CC = kornia.augmentation.CenterCrop((64,64))
    
    net = net.eval()

    correct = torch.tensor(0)
    dataiter = iter(testloader)
    total = torch.tensor(0)
    # total = 0 
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs_left,inputs_right, labels,info = data
            inputs = torch.cat((inputs_left,inputs_right),dim=1)
#             print(inputs_left.shape)
#             print(inputs_right.shape)
#             inputs = torch.cat((inputs_left,inputs_right),dim=1)
#             print(inputs.shape)
            
            inputs = inputs.cuda() 
            indices = torch.argwhere(labels!=5)
            labels = labels.cuda()
            inputs_temp = CC(inputs)
            TT = time.time()
            all_outs = net(inputs_temp)
            print((time.time()-TT)/inputs.shape[0])
            
            for temp in range(10):
                inputs_temp = RC(inputs)
                all_outs = all_outs + net(inputs_temp)
            
            predicted = torch.argmax(all_outs,1)
            correct = correct + torch.sum(predicted[indices] == labels[indices])
            total = total+ len(indices)
    accuracy = float(correct) / float(total)
    print(total)
    return accuracy


# -

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
                all_outs = net(inputs)
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
            all_outs = net(inputs)
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
        output = model(datap)
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
        
        output = model(perturbed_data)
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

# sys.path.append('../models') 
# from vgg import *
import time 
from vgg_norb import * 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# +
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
    corruptions = ['general']
    atanh_convert = 0 

    gc.collect()
    torch.cuda.empty_cache()
    
    dataset_name = "NORB"
    batch_size = 100
    init_rate = 0.0005
    init_rate_crank = 0.01
    labelnoise = 0
    input_noise = 0 
    input_noise_array = [0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.25,0.3,0.35]
    epsilons = [ .01]
    srn_epsilons = 0.00001
    # test_transforms = ['scale','translate','rotate','']
    
    
    step_size = 10
    gamma_learning = 0.8
    total_epoch = 300
    total_epoch_crank = 100
    decay_normal = 0    
    decay_regress = 0
    decay_errors = 0
    dropping = 0 
    
    decay_normal_crank = 0 
    
    layers = [25,50]
    kernels = [5,5]
    layers_crank = []
    
    training_size = 50000
    mode = 'regress_batch'
    use_bn = True
    rescale = 1.0
    rank_convert = False
    input_channels = 2
    global DEGREES
    DEGREE = [1,8]
    alpha=0.2
    
    net = Shallow_GMUCNN(input_channels,layers,kernels,srn_epsilons,use_bn=use_bn,dropping=dropping)
#     net = VGG('VGG16-SRN')
#     dataset,dataset_test,my_dataset,my_dataset_test,trainloader,testloader,testloader_adversary = load_data_and_generators(dataset_name,training_size,rescale,rank_convert,atanh_convert,labelnoise)
    trainloader,testloader,testloader_norb = load_data_and_generators(dataset_name,training_size,rescale,rank_convert,atanh_convert,labelnoise)

    
    # print('Mode:', mode)
    # # if mode == 'regress_batch':
    net = Shallow_GMUCNN(input_channels,layers,kernels,srn_epsilons,use_bn=use_bn,dropping=dropping)
#     net.load_state_dict(torch.load('./stereo_Shallow_GMUCNN_norb_1slices_1degree_5kernel_avgpool_64input_nopadding_moremoreepochs.h5', weights_only=True))
#     net.load_state_dict(torch.load('./stereo_Shallow_GMUCNN_norb_1slices_2degree_5kernel_avgpool_64input_nopadding.h5', weights_only=True))
    net.load_state_dict(torch.load('./stereo_Shallow_GMUCNN_norb_3slices_5kernel_avgtpool_64input_nopadding.h5', weights_only=True))

    print(count_parameters(net))
    
#     TT = time.time()
    accuracy = test_network_pairs(net, testloader, 0)
    
    print('Test Accuracy GMU-CNN (Normal):', accuracy)
    
    
    net_normal = Shallow_CNN(input_channels,layers,kernels,srn_epsilons,use_bn=use_bn,dropping=0)
    net_normal.load_state_dict(torch.load('./stereo_Shallow_CNN_norb_64dim_nopadding_5kernel.h5', weights_only=True))
    
    TT = time.time()
    accuracy = test_network_pairs(net_normal, testloader, 0)

    print('Test Accuracy CNN (Normal):', accuracy)
    
#    
# -


