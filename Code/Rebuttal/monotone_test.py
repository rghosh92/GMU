# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 00:05:08 2026

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:53:43 2026

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

# +
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
# torch.set_float32_matmul_precision("high")

import kornia.augmentation as K
import kornia.color

# Define Kornia augmentations

sys.path.append('./models') 
from vgg import *
from svhn_nets import * 


import timm




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def extract_dataset_features(model, dataloader, feature_dict, device='cuda'):
    model.eval()
    model.to(device)
    
    all_features = {'layer1': [], 'layer2': [], 'layer3': []}
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs) 
            
            for layer in ['layer1', 'layer2', 'layer3']:
                # Read from the passed feature_dict!
                all_features[layer].append(feature_dict[layer]) 
            
            all_labels.append(labels.cpu().numpy())
            
    for layer in all_features:
        all_features[layer] = np.vstack(all_features[layer])
    all_labels = np.concatenate(all_labels)
    
    return all_features, all_labels



# def get_features(storage_dict, name):
#     def hook(model, input, output):
#         pooled = F.adaptive_avg_pool2d(output, (1, 1))
#         storage_dict[name] = pooled.flatten(start_dim=1).detach().cpu().numpy()
#     return hook


def get_features(storage_dict, name):
    def hook(model, input, output):
        # Flatten the spatial dimensions: (Batch, Channels, H, W) -> (Batch, Channels * H * W)
        storage_dict[name] = output.detach().view(output.size(0), -1).cpu().numpy()
    return hook

def get_gmu_matched_features(storage_dict, name):
    def hook(model, input, output):
        # Apply MaxPool2d (kernel_size=2, stride=2) to reduce 32x32 -> 16x16
        pooled = F.max_pool2d(output, kernel_size=2, stride=2)
        storage_dict[name] = pooled.flatten(start_dim=1).detach().cpu().numpy()
    return hook


# def get_features(name):
#     def hook(model, input, output):
#         # Global Average Pooling over H and W: (B, C, H, W) -> (B, C, 1, 1)
#         pooled = F.adaptive_avg_pool2d(output, (1, 1))
#         # Flatten spatial dimensions: (B, C, 1, 1) -> (B, C)
#         features[name] = pooled.flatten(start_dim=1).detach().cpu().numpy()
#     return hook

# -


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




# +
import itertools as it



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



def train_network_normal(net,trainloader,testloader,  init_rate,epochs,weight_decay):
    
    transform = torch.nn.Sequential(
        K.RandomRotation(degrees=30),  # Equivalent to rotation_range=8
#         K.RandomResizedCrop(size=(32, 32), scale=(0.95, 1.05)),  # Equivalent to zoom_range=[0.95, 1.05]
#         K.RandomAffine(degrees=0, shear=0.15),  # Equivalent to shear_range=0.15
        K.RandomAffine(degrees=0, translate=(0, 0.10)),  # Equivalent to height_shift_range=0.10
        )
    
    net = net.cuda()
    net = net.train()
#     optimizer = optim.Adam(net.parameters(), lr=init_rate, weight_decay=0)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Define Cosine Annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.0001)


    criterion = nn.CrossEntropyLoss()
    
    init_epoch = 0
    all_train_losses = []
    bazinga = 0 
    train_loss_min = 9999
    for epoch in range(epochs):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # scheduler.step()
        # print('epoch: ' + str(epoch))
        train_loss = []
        loss_weights = [] 
       
        print(epoch)
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            inputs = transform(inputs)
            
            optimizer.zero_grad()
            
            # print(3)
            # print(inputs.dtype)
            allouts = net(inputs)
            loss = criterion(allouts, labels.long()) #+ net.decay_errors*torch.mean(-x_errs)
            loss.backward()
            train_loss.append(loss.item())
            loss_weights.append(len(labels))
            
          
            optimizer.step()
        
        scheduler.step()
        
#         for g in optimizer.param_groups:
#             g['lr'] = init_rate* (0.5 ** (epoch // 50))

        
        all_train_losses.append(np.average(np.array(train_loss),weights=np.array(loss_weights)))
        print(all_train_losses[-1])
        if all_train_losses[-1] < train_loss_min:
            train_loss_min = copy(all_train_losses[-1])
            net_best = deepcopy(net)
        
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


class mini_dataset():
    def __init__(self):
        data = 0 
        targets = 0 

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


def load_data_and_generators(dataset_name, training_size, scale, rank_convert, labelnoise):
    
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
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        dataset_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
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
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        dataset.data = torch.permute(torch.tensor(dataset.data), (0, 3, 1, 2)).float() / 255.0
        dataset.targets = torch.tensor(dataset.targets)
        dataset_test.data = torch.permute(torch.tensor(dataset_test.data), (0, 3, 1, 2)).float() / 255.0
        dataset_test.targets = torch.tensor(dataset_test.targets)

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

    elif dataset_name == 'CUB200':
        dataset = Cub2011(root='./data', train=True, transform=transform_train, download=True)
        dataset_test = Cub2011(root='./data', train=False, transform=transform_test, download=True)
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported!")
        
    
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


# def rank_convert_data(data,**kwargs):
    
#     for i in range(len(data)):
#         temp,data[i] = torch.unique(data[i],return_inverse=True)
#         # dataset.data = dataset.data
    
#     return data

from kornia.color import *


# def test_network(net, testloader):
#     net = net.eval()
    
#     # net = net.train()
#     for layer in net.modules():
#         if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
#             layer.momentum = 0.9
            
#     correct = torch.tensor(0)
#     dataiter = iter(testloader)
#     total_labels = torch.tensor(0)
#     # total = 0 
#     num_augmentations = 5  # Number of augmented versions per image

#     transform = torch.nn.Sequential(
#         K.CenterCrop((32, 24)),
#         K.Resize((28, 28)),
        
#         # K.RandomAffine(degrees=0, translate=(0, 0.1)),
#         # K.RandomRotation(degrees=15),
#     )
#     num_classes = 10  # Get number of output neurons from the model

#     num_augmentations = 1  # Number of augmentations per image

#     correct = torch.tensor(0)
#     total_labels = torch.tensor(0)
    
#     for inputs, labels in testloader:
#         inputs = rgb_to_grayscale(inputs).cuda()
#         labels = labels.cuda()
    
#         all_probs = torch.zeros((inputs.shape[0], num_augmentations, num_classes)).cuda()
    
#         for i in range(num_augmentations):

#             aug_inputs = transform(inputs)  # Apply augmentation
#             all_outs, _ = net(aug_inputs)
#             all_probs[:, i, :] = torch.nn.functional.softmax(all_outs, dim=1)  # Convert logits to probabilities
    
#         avg_probs = torch.mean(all_probs, dim=1)  # Average probabilities across augmentations
#         predicted = torch.argmax(avg_probs, dim=1)
    
#         correct += torch.sum(predicted == labels)
#         total_labels += labels.shape[0]
    
#     accuracy = float(correct.item()) / float(total_labels.item())
#     return accuracy




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



def extract_first_layer(model, dataloader, feature_dict, layer_key='conv1', device='cuda'):
    model.eval()
    model.to(device)
    
    num_samples = len(dataloader.dataset)
    
    # 1. Warmup pass with 1 sample to get exact output feature dimensions
    sample_inputs, _ = next(iter(dataloader))
    with torch.no_grad():
        _ = model(sample_inputs[:1].to(device))
    num_features = feature_dict[layer_key].shape[1]
    
    # 2. Pre-allocate continuous memory blocks directly
    all_feats = np.empty((num_samples, num_features), dtype=np.float32)
    all_labels = np.empty(num_samples, dtype=np.int64)
    
    # 3. Fill directly in-place (no list growth, no vstack overhead)
    ptr = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            _ = model(inputs) # Triggers hook
            
            all_feats[ptr : ptr + batch_size] = feature_dict[layer_key]
            all_labels[ptr : ptr + batch_size] = labels.cpu().numpy()
            
            ptr += batch_size
            
    return all_feats, all_labels


class MonotonicPowerTransform:
    def __init__(self, gamma=2.0):
        """
        Monotonic power scaling f(x) = x^gamma.
        Preserves ordering on [0, 1] while non-linearly changing contrast.
        """
        self.gamma = gamma

    def __call__(self, img):
        # Clamp to [0, 1] safety bound then raise to power
        return torch.pow(torch.clamp(img, 0.0, 1.0), self.gamma)


class MonotonicRankTransform:
    def __init__(self):
        """
        Maps pixel values to their percentile ranks [0, 1] per channel.
        Strictly preserves x < y => f(x) < f(y).
        """
    def __call__(self, img):
        c, h, w = img.shape
        img_flat = img.view(c, -1)
        
        # Compute pixel ranks along spatial dimension
        sorted_indices = torch.argsort(img_flat, dim=1)
        ranks = torch.zeros_like(img_flat)
        
        # Convert index order into normalized rank values [0.0, 1.0]
        rank_grid = torch.linspace(0.0, 1.0, steps=h*w, device=img.device)
        for channel in range(c):
            ranks[channel, sorted_indices[channel]] = rank_grid
            
        return ranks.view(c, h, w)
    
def evaluate_monotonic_invariance(base_feats, mono_feats):
    # Normalize vectors for Cosine Similarity
    base_norm = base_feats / (np.linalg.norm(base_feats, axis=1, keepdims=True) + 1e-8)
    mono_norm = mono_feats / (np.linalg.norm(mono_feats, axis=1, keepdims=True) + 1e-8)
    
    cos_sim = np.mean(np.sum(base_norm * mono_norm, axis=1))
    
    # Relative norm change
    rel_distortion = np.mean(
        np.linalg.norm(base_feats - mono_feats, axis=1) / 
        (np.linalg.norm(base_feats, axis=1) + 1e-8)
    )
    return cos_sim, rel_distortion


def evaluate_and_plot_monotonic_invariance(base_x_cnn, mono_x_cnn, base_x_gmu, mono_x_gmu):
    """
    Computes per-sample cosine similarities and relative distortions,
    prints full distribution stats (Mean, Median, Std, P95), and plots distributions.
    """
    # Helper to compute per-sample metrics
    def compute_per_sample_metrics(base_feats, mono_feats):
        # Flatten spatial dimensions if present: [N, C, H, W] -> [N, C*H*W]
        if base_feats.ndim > 2:
            base_feats = base_feats.reshape(base_feats.shape[0], -1)
            mono_feats = mono_feats.reshape(mono_feats.shape[0], -1)
            
        # Per-sample Cosine Similarity: [N]
        print(base_feats.shape)
        base_norm = base_feats / (np.linalg.norm(base_feats, axis=1, keepdims=True) + 1e-8)
        mono_norm = mono_feats / (np.linalg.norm(mono_feats, axis=1, keepdims=True) + 1e-8)
        per_sample_cos = np.sum(base_norm * mono_norm, axis=1)
        
        # Per-sample Relative Distortion: [N]
        diff_norm = np.linalg.norm(base_feats - mono_feats, axis=1)
        base_mag = np.linalg.norm(base_feats, axis=1) + 1e-8
        per_sample_dist = diff_norm / base_mag
        
        return per_sample_cos, per_sample_dist

    # 1. Compute per-sample arrays
    cnn_cos, cnn_dist = compute_per_sample_metrics(base_x_cnn, mono_x_cnn)
    gmu_cos, gmu_dist = compute_per_sample_metrics(base_x_gmu, mono_x_gmu)

    # 2. Print Detailed Distribution Statistics
    def print_stats(name, cos_arr, dist_arr):
        print(f"\n=== {name} Monotonic Invariance Statistics ===")
        print(f"Cosine Sim  -> Mean: {np.mean(cos_arr):.4f} | Median: {np.median(cos_arr):.4f} | Std: {np.std(cos_arr):.4f} | Min: {np.min(cos_arr):.4f}")
        print(f"Distortion  -> Mean: {np.mean(dist_arr):.4f} | Median: {np.median(dist_arr):.4f} | Std: {np.std(dist_arr):.4f} | Max (P95): {np.percentile(dist_arr, 95):.4f}")

    print_stats("CNN", cnn_cos, cnn_dist)
    print_stats("GMU", gmu_cos, gmu_dist)

    # 3. Create Clean 2-Panel Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Activation Shift Under Monotonic Transformation: CNN vs. GMU", fontsize=14, fontweight='bold', y=0.98)

    # Panel 1: Cosine Similarity Histogram/Density
    bins = np.linspace(min(np.min(cnn_cos), np.min(gmu_cos)), 1.0, 30)
    axes[0].hist(cnn_cos, bins=bins, alpha=0.6, label=f"CNN (Mean: {np.mean(cnn_cos):.3f})", color='crimson', density=True)
    axes[0].hist(gmu_cos, bins=bins, alpha=0.6, label=f"GMU (Mean: {np.mean(gmu_cos):.3f})", color='navy', density=True)
    axes[0].set_title("Cosine Similarity Distribution", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Cosine Similarity (Higher = More Invariant)")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Relative Distortion Distribution (Boxplot for Outlier / Spread Visibility)
    # Panel 2: Relative Distortion Distribution
    # Added showfliers=False to prevent extreme CNN outliers from stretching the y-axis
    # and squishing the boxes.
    box = axes[1].boxplot(
        [cnn_dist, gmu_dist], 
        labels=['CNN', 'GMU'], 
        patch_artist=True, 
        widths=0.4, 
        showfliers=False  # <-- THIS IS THE KEY FIX
    )
    
    colors = ['crimson', 'navy']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    # Optional: We can still dynamically set a slight top margin based on the highest whisker
    # to ensure it looks perfectly framed.
    axes[1].set_title("Relative Feature Distortion ||Δx|| / ||x||", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Relative Distortion (Lower = More Invariant)")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

    return (cnn_cos, cnn_dist), (gmu_cos, gmu_dist)

def monotonic_power_transform_tensor(img, gamma=2.5):
    """Applies contrast shift directly to the normalized float tensor."""
    return torch.pow(torch.clamp(img, 0.0, 1.0), gamma)

def monotonic_rank_transform_tensor(img_tensor):
    """
    Replaces each pixel value with its spatial rank (0 to 1) within its channel.
    Input shape: [N, C, H, W]
    """
    N, C, H, W = img_tensor.shape
    # Flatten spatial dimensions to compute ranks across the whole image
    flat_img = img_tensor.reshape(N, C, -1)
    
    # Applying argsort twice gives the ordinal rank of each element
    ranks = torch.argsort(torch.argsort(flat_img, dim=-1), dim=-1).float()
    
    # Normalize ranks to [0.0, 1.0]
    ranks_normalized = ranks / (H * W - 1)
    
    return ranks_normalized.reshape(N, C, H, W)


if __name__ == "__main__":
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    

    gc.collect()
    torch.cuda.empty_cache()
    
    dataset_name = "SVHN"
    batch_size = 20
    init_rate = 0.0005
    init_rate_crank = 0.01
    labelnoise = 0
    input_noise = 0 
    input_noise_array = [0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.25,0.3,0.35]
    epsilons = [ .05]
    srn_epsilons = [0.01,0.00001,0]

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

    training_size = 10000
    mode = 'regress_batch'
    use_bn = True
    rescale = 1.0
    rank_convert = False
    input_channels = 1
    global DEGREES
    DEGREE = [1,8]
    alpha=0.2
    
    # net = VGG('VGG16-GMU',num_slices=3)
    dataset,dataset_test,trainloader,testloader = load_data_and_generators(dataset_name,training_size,rescale,rank_convert,labelnoise)
    
    
    resnet18_gmu = timm.create_model("resnet18", pretrained=False)


    # Replace First Convolutional Layer with GMULayer
    resnet18_gmu.conv1 = GMULayer(3, 64, 7, padding=3, epsilon=0.0001, num_slices=3, degree=1)


    resnet18_gmu = resnet18_gmu.cuda()
    
    resnet18 = timm.create_model("resnet18", pretrained=False)

    
    resnet18_gmu.load_state_dict(torch.load('Resnet18_SVHN_3slices_cosine_lr_augset_0.1sgd.pth',weights_only=True))
    resnet18.load_state_dict(torch.load('Resnet18_SVHN_normal_cosine_lr_augset_0.1sgd.pth',weights_only=True))

    resnet18.eval()
    resnet18_gmu.eval()
    
    
    features_normal = {}
    features_gmu = {}
    handle_norm = resnet18.act1.register_forward_hook(get_features(features_normal, 'conv1'))
    handle_gmu = resnet18_gmu.act1.register_forward_hook(get_gmu_matched_features(features_gmu, 'conv1'))        

        # Initialize Monotonic Transformations
    # -------------------------------------------------------------------------
    # 1. FIXED DATA PREPARATION (Fixes data overwrite bug)
    # -------------------------------------------------------------------------
    # Import or define the transform directly to apply on the tensor
    
    print("Preparing monotonic dataset...")
    # Load dataset normally to get the correct count and base data
    dataset_test_mono = SVHN(root='./data', split='test', download=True)

    # 1. Take the base raw [0-255] data -> [N, C, H, W] float [0, 1]
    # (SVHN is already NCHW raw)
    baseline_raw_data = torch.tensor(dataset_test_mono.data).float() / 255.0

    # 2. Apply monotonic transform IN-PLACE directly on the tensor
    gamma_val = 2.5
    mono_data_transformed = monotonic_power_transform_tensor(baseline_raw_data, gamma=gamma_val)

    # 3. Overwrite dataset array and set identity transform
    # The dataloader will now serve the pre-transformed data.
    dataset_test_mono.data = mono_data_transformed
    dataset_test_mono.transform = lambda x: x # Identity (already tensor)

    testloader_mono = torch.utils.data.DataLoader(
        dataset_test_mono,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 2. Extract baseline vs monotonic activations
    print("Extracting CNN baseline vs. monotonic features...")
    base_x_cnn, _ = extract_first_layer(resnet18, testloader, features_normal, layer_key='conv1')
    mono_x_cnn, _ = extract_first_layer(resnet18, testloader_mono, features_normal, layer_key='conv1')
    
    print("Extracting GMU baseline vs. monotonic features...")
    base_x_gmu, _ = extract_first_layer(resnet18_gmu, testloader, features_gmu, layer_key='conv1')
    mono_x_gmu, _ = extract_first_layer(resnet18_gmu, testloader_mono, features_gmu, layer_key='conv1')
    
    # 3. Compute Metrics
    
    cnn_stats, gmu_stats = evaluate_and_plot_monotonic_invariance(
        base_x_cnn, mono_x_cnn, 
        base_x_gmu, mono_x_gmu
    )
    
    
    # =========================================================================
    # RANK TRANSFORM EVALUATION
    # =========================================================================
    print("\nPreparing Rank (Histogram Equalized) dataset...")
    
    dataset_test_rank = SVHN(root='./data', split='test', download=True)
    
    # Apply rank transform IN-PLACE directly on the tensor
    rank_data_transformed = monotonic_rank_transform_tensor(baseline_raw_data)
    
    dataset_test_rank.data = rank_data_transformed
    dataset_test_rank.transform = lambda x: x # Identity
    
    testloader_rank = torch.utils.data.DataLoader(
        dataset_test_rank,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Extract rank features
    print("Extracting CNN baseline vs. rank features...")
    mono_rank_cnn, _ = extract_first_layer(resnet18, testloader_rank, features_normal, layer_key='conv1')
    
    print("Extracting GMU baseline vs. rank features...")
    mono_rank_gmu, _ = extract_first_layer(resnet18_gmu, testloader_rank, features_gmu, layer_key='conv1')

    # Evaluate and Plot (Re-using the plotting function from earlier)
    print("\n" + "="*50)
    print("EVALUATING MONOTONIC RANK TRANSFORM")
    print("="*50)
    cnn_rank_stats, gmu_rank_stats = evaluate_and_plot_monotonic_invariance(
        base_x_cnn, mono_rank_cnn, 
        base_x_gmu, mono_rank_gmu
    )
    
    
    # cnn_cos, cnn_dist = evaluate_monotonic_invariance(base_x_cnn, mono_x_cnn)
    # gmu_cos, gmu_dist = evaluate_monotonic_invariance(base_x_gmu, mono_x_gmu)
    
    # print("\n--- RESULTS FOR MONOTONIC TRANSFORM ---")
    # print(f"CNN  -> Cosine Sim: {cnn_cos:.4f} | Relative Distortion: {cnn_dist:.4f}")
    # print(f"GMU  -> Cosine Sim: {gmu_cos:.4f} | Relative Distortion: {gmu_dist:.4f}")

  