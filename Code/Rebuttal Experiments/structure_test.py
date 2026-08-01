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


def run_linear_probe(train_x, train_y, test_x, test_y, layer_name, num_classes=10, epochs=100, lr=0.01, batch_size=200):
    # Ensure inputs are PyTorch Tensors on GPU
    if not isinstance(train_x, torch.Tensor):
        train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
        train_y = torch.tensor(train_y, dtype=torch.long).cuda()
        test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
        test_y = torch.tensor(test_y, dtype=torch.long).cuda()
    else:
        train_x, train_y = train_x.cuda(), train_y.cuda()
        test_x, test_y = test_x.cuda(), test_y.cuda()
    
    g_cuda = torch.Generator(device='cuda')
    
    eps = 1e-8
    x_min = train_x.min(dim=0, keepdim=True).values
    x_max = train_x.max(dim=0, keepdim=True).values

    train_x = (train_x - x_min) / (x_max - x_min + eps)
    test_x  = (test_x - x_min)  / (x_max - x_min + eps)  # Uses train min/max to prevent data leak!
    
    
    # Now TensorDataset receives native PyTorch tensors
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, generator=g_cuda)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False, generator=g_cuda)
    
    in_features = train_x.shape[1]
    probe = nn.Linear(in_features, num_classes).cuda()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # --- Training ---
    probe.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(probe(bx), by)
            loss.backward()
            optimizer.step()

    # --- Evaluation ---
    probe.eval()
    def evaluate(loader):
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in loader:
                preds = probe(bx).argmax(dim=1)
                correct += (preds == by).sum().item()
                total += by.shape[0]
        return correct / total

    train_acc = evaluate(train_loader)
    test_acc = evaluate(test_loader)

    print(f"[{layer_name}] Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
    return train_acc, test_acc
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


def load_data_and_generators(dataset_name, training_size, scale, rank_convert,  labelnoise):
    
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


def test_network_corrupted_CIFAR_TTA(net_orig, corruptions):
    accuracy_list = [] 
    for corruption in corruptions:
        net = deepcopy(net_orig)
        net = net.train()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                layer.momentum = 0.9  # Update momentum for faster adaptation
        # print(corruption)
        gc.collect()
        torch.cuda.empty_cache()
        data_test = np.load('./CIFAR-10-C/'+ corruption +'.npy')
        labels_test = np.load('./CIFAR-10-C/labels.npy')
        # data_test = (torch.from_numpy(data_test)).view(data_test.shape[0],
                                                     # int(data_test.size/data_test.shape[0]),1,1)
        labels_test = torch.from_numpy(labels_test)
        data_test = (torch.from_numpy(data_test).permute(0,3,1,2)).float()/(255.0)
        # if atan_convert>0:
        #     data_test = map_inf(data_test, atan_convert)
        
        my_dataset_test = Dataset(dataset_name, data_test.cuda(), labels_test.cuda())

        testloader = torch.utils.data.DataLoader(my_dataset_test, batch_size=200,
                                              shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)
        
        
        correct = torch.tensor(0)
        dataiter = iter(testloader)
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                # get the inputs
                inputs, labels = data
                _ = net(inputs)
                net = net.eval()
                
                all_outs = net(inputs)
                predicted = torch.argmax(all_outs,1)

                correct = correct + torch.sum(predicted == labels)
                net = net.train()
        accuracy = float(correct) / float(len(labels_test))
        accuracy_list.append(accuracy)
        print("Corruption:",corruption, " Accuracy: ", accuracy)
    
    print("Mean accuracy:",np.mean(accuracy_list))
    return 0
import kornia.color as K_color


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



# -
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


# +
class Net_vanilla_CNN_convert3(nn.Module):
    def __init__(self,input_channels,layers,kernels,epsilons = [0.0001,0.0001,0.0001],decay_regress=0,decay_errors = 0, classes = 10,use_bn = True,dropping=0,poly_order_init=5):
        super(Net_vanilla_CNN_convert3, self).__init__()
        
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
        self.srn1 = SRNLayer(input_channels, 64, 3,padding=self.padding[0],epsilon = epsilons[0],num_slices=3,degree=1 )
        
        
        
        if self.mode =='normal':
            self.conv1 = nn.Conv2d(1,64,5, padding=2)
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
        
        
    
        
    def forward(self, x,bazinga=1):
       
        
        if self.mode == 'normal':
            x = self.srn1(x)
            x = self.feat_net(x)
            x_errs = x
                    
             
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
       
        xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)
        
       
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm,x_errs
    
  
    
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
        
        #
        
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


# +

import cv2



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


class FixedPixelShuffleTransform:
    def __init__(self, image_size=(32, 32)):
        """
        Creates a single, fixed random permutation of pixel indices.
        Applying this to all images destroys spatial structure while 
        keeping the exact same global pixel distribution.
        """
        self.h, self.w = image_size
        # Generate one fixed permutation for the entire dataset
        self.perm_idx = torch.randperm(self.h * self.w)

    def __call__(self, img):
        # img shape: [C, H, W]
        c = img.shape[0]
        
        # Flatten the spatial dimensions: [C, H*W]
        img_flat = img.view(c, -1)
        
        # Apply the fixed permutation across all channels identically
        # (This keeps the RGB values of a single pixel together)
        img_shuffled = img_flat[:, self.perm_idx].view(c, self.h, self.w)
        
        return img_shuffled
    


def evaluate_activation_shifts(model, base_loader, perturbed_loader, feature_dict, layer_key='conv1'):
    """
    Measures the degree of activation reduction, energy collapse, and sparsity 
    changes when structural priors are destroyed.
    """
    print(f"Extracting baseline features...")
    base_x, _ = extract_first_layer(model, base_loader, feature_dict, layer_key=layer_key)
    
    print(f"Extracting perturbed features...")
    pert_x, _ = extract_first_layer(model, perturbed_loader, feature_dict, layer_key=layer_key)
    
    # 1. Mean Absolute Response Magnitude
    base_mag = np.mean(base_x)
    pert_mag = np.mean(pert_x)
    mag_ratio = pert_mag / (base_mag + 1e-8)  # < 1.0 means suppression
    
    # 2. Frobenius Norm / Energy Ratio
    base_energy = np.mean(np.square(base_x))
    pert_energy = np.mean(np.square(pert_x))
    energy_ratio = pert_energy / (base_energy + 1e-8)
    
    # Decibels of energy loss (negative means energy drop)
    db_drop = 10 * np.log10(energy_ratio + 1e-12)
    
    # 3. Sparsity / Silent Activations (|x| < 1e-4)
    # Measures if neurons are effectively deactivated
    base_sparsity = np.mean(base_x < 1e-4)
    pert_sparsity = np.mean(pert_x < 1e-4)
    
    print(f"\n--- Activation Reduction Report [{layer_key}] ---")
    print(f"  Base Mean Magnitude : {base_mag:.4f}  -->  Shuffled: {pert_mag:.4f}")
    print(f"  Magnitude Retention : {mag_ratio * 100:.2f}% of baseline")
    print(f"  Energy Retention    : {energy_ratio * 100:.2f}% ({db_drop:.2f} dB drop)")
    print(f"  Inactive Neurons    : Base {base_sparsity * 100:.1f}% --> Shuffled {pert_sparsity * 100:.1f}%")
    
    return {
        "mag_ratio": mag_ratio,
        "energy_ratio": energy_ratio,
        "db_drop": db_drop,
        "base_mag": base_mag,
        "pert_mag": pert_mag,
        "base_sparsity": base_sparsity,
        "pert_sparsity": pert_sparsity
    }

def compute_spatial_center_diff(weights, is_gmu=False):
    """
    Computes the mean absolute difference of spatial filter weights relative to their center pixel.

    weights shape:
        CNN: [out_channels, in_channels, K, K]
        GMU: [out_channels, in_channels, K, K, num_slices]
    """
    # 1. Collapse non-spatial dimensions via L2 norm to get spatial magnitude per filter: [Out, K, K]
    if is_gmu:
        # Norm across input channels (dim 1) and slices (dim 4)
        spatial_mag = torch.linalg.norm(weights, dim=(1, 4))
    else:
        # Norm across input channels (dim 1)
        spatial_mag = torch.linalg.norm(weights, dim=1)

    K = spatial_mag.shape[-1]
    center_idx = K // 2  # (3, 3) for a 7x7 kernel

    # 2. Extract center magnitude per filter: shape [Out, 1, 1]
    center_val = spatial_mag[:, center_idx, center_idx].unsqueeze(-1).unsqueeze(-1)

    # 3. Absolute difference from center across all spatial locations: [Out, K, K]
    center_diff = torch.abs(spatial_mag - center_val)

    # 4. Average across all output channels/filters to get a single 2D map: [K, K]
    mean_diff_map = torch.mean(center_diff, dim=0).detach().cpu().numpy()

    return mean_diff_map

def plot_and_quantify_spatial_structure(resnet_normal, resnet_gmu):
    # Extract weight tensors
    cnn_w = resnet_normal.conv1.weight
    gmu_w = resnet_gmu.conv1.weights

    # Compute 2D center-difference maps
    cnn_diff = compute_spatial_center_diff(cnn_w, is_gmu=False)
    gmu_diff = compute_spatial_center_diff(gmu_w, is_gmu=True)

    # Quantitative Metric: Spatial Total Variation (lower TV = smoother spatial transitions)
    def spatial_total_variation(img):
        diff_x = np.abs(img[1:, :] - img[:-1, :])
        diff_y = np.abs(img[:, 1:] - img[:, :-1])
        return np.mean(diff_x) + np.mean(diff_y)

    cnn_tv = spatial_total_variation(cnn_diff)
    gmu_tv = spatial_total_variation(gmu_diff)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(cnn_diff, cmap='magma')
    axes[0].set_title(f"CNN Conv1 Center-Diff Map\nSpatial Smoothness (TV): {cnn_tv:.4f}")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(gmu_diff, cmap='magma')
    axes[1].set_title(f"GMU Conv1 Center-Diff Map\nSpatial Smoothness (TV): {gmu_tv:.4f}")
    fig.colorbar(im1, ax=axes[1])

    for ax in axes:
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xlabel("Kernel X")
        ax.set_ylabel("Kernel Y")

    plt.suptitle("Average Weight Magnitude Difference from Center Pixel |W(x,y) - W(center)|", fontsize=12)
    plt.tight_layout()
    plt.show()
    
def compute_normalized_spatial_roughness(weights, is_gmu=False):
    """Computes scale-invariant spatial roughness:

    E[(w_i - w_{N(i)})^2] / Var(w)

    Lower value = spatially smoother / higher neighbor correlation.
    """
    # 1. Collapse non-spatial dimensions via L2 norm to get 2D spatial magnitude maps per filter: [Out, K, K]
    if is_gmu:
        # Norm across input channels (dim 1) and slices (dim 4)
        spatial_mag = torch.linalg.norm(weights, dim=(1, 4))
    else:
        # Norm across input channels (dim 1)
        spatial_mag = torch.linalg.norm(weights, dim=1)

    # Convert to float numpy array for spatial diffs
    mag_maps = spatial_mag.detach().cpu().numpy()  # Shape: [Out_channels, K, K]

    ratios = []

    for img in mag_maps:
        # Compute horizontal and vertical neighbor differences
        diff_x = (img[:, 1:] - img[:, :-1]) ** 2  # Horizontal neighbors
        diff_y = (img[1:, :] - img[:-1, :]) ** 2  # Vertical neighbors

        # Total Mean Squared Neighbor Difference (E[(w_i - w_{N(i)})^2])
        mean_neighbor_diff = np.mean(diff_x) + np.mean(diff_y)

        # Variance of spatial weights within this filter
        var_w = np.var(img)

        # Avoid division by zero for flat/zero filters
        if var_w > 1e-8:
            ratios.append(mean_neighbor_diff / var_w)

    # Return average scale-invariant roughness across all output channels
    return np.mean(ratios)

    
def visualize_spatial_weights(resnet_normal, resnet_gmu, num_filters=16):
    """
    Computes the spatial norm of the weights and plots them side-by-side.
    """
    # CNN Weights: [64, 3, 7, 7]
    # We take the norm over the input channel (dim=1) to get [64, 7, 7]
    cnn_weights = resnet_normal.conv1.weight.detach().cpu()
    cnn_spatial = torch.linalg.norm(cnn_weights, dim=1) 
    
    # GMU Weights: [64, 3, 7, 7, num_slices]
    # We take the norm over input channels (dim=1) AND slices (dim=4) to get [64, 7, 7]
    gmu_weights = resnet_gmu.conv1.weights.detach().cpu()
    gmu_spatial = torch.linalg.norm(gmu_weights, dim=(1, 4))
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle("Spatial Contiguity of Learned Weights: CNN vs GMU", fontsize=16)
    
    for i in range(num_filters):
        # Plot CNN Filter
        ax_cnn = axes[i // 4, (i % 4) * 2]
        im_cnn = ax_cnn.imshow(cnn_spatial[i].numpy(), cmap='viridis')
        ax_cnn.axis('off')
        if i < 4: ax_cnn.set_title("CNN Filter")
        
        # Plot GMU Filter
        ax_gmu = axes[i // 4, (i % 4) * 2 + 1]
        im_gmu = ax_gmu.imshow(gmu_spatial[i].numpy(), cmap='plasma')
        ax_gmu.axis('off')
        if i < 4: ax_gmu.set_title("GMU Filter")
        
    plt.tight_layout()
    plt.show()



def plot_activation_distributions(base_cnn, pert_cnn, base_gmu, pert_gmu):
    """
    Plots histograms of activation values to verify feature shifts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # CNN Histogram
    axes[0].hist(base_cnn.flatten(), bins=100, alpha=0.5, label='CNN Baseline', color='blue', density=True)
    axes[0].hist(pert_cnn.flatten(), bins=100, alpha=0.5, label='CNN Shuffled', color='cyan', density=True)
    axes[0].set_title("CNN First Layer Activation Distribution")
    axes[0].set_xlabel("Activation Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    
    # GMU Histogram
    axes[1].hist(base_gmu.flatten(), bins=100, alpha=0.5, label='GMU Baseline', color='red', density=True)
    axes[1].hist(pert_gmu.flatten(), bins=100, alpha=0.5, label='GMU Shuffled', color='orange', density=True)
    axes[1].set_title("GMU First Layer Activation Distribution")
    axes[1].set_xlabel("Activation Value")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()




def plot_activation_collapse(cnn_results, gmu_results):
    labels = ['Normal Conv1', 'GMU Conv1']
    base_mags = [cnn_results['base_mag'], gmu_results['base_mag']]
    pert_mags = [cnn_results['pert_mag'], gmu_results['pert_mag']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 5))
    rects1 = ax.bar(x - width/2, base_mags, width, label='Clean Input', color='steelblue')
    rects2 = ax.bar(x + width/2, pert_mags, width, label='Pixel-Shuffled', color='crimson')
    
    ax.set_ylabel('Mean Activation')
    ax.set_title('Activation Collapse Under Pixel Shuffling')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Annotate percentage drop on top of bars
    for i in range(2):
        retention = [cnn_results['mag_ratio'], gmu_results['mag_ratio']][i] * 100
        drop = 100 - retention
        ax.annotate(f"-{drop:.1f}%",
                    xy=(x[i] + width/2, pert_mags[i]),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
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
    
    # Initialize the transform once so the permutation is fixed
    pixel_shuffle = FixedPixelShuffleTransform(image_size=(32, 32))
    
            # 1. Load the dataset normally
        # 1. Load the dataset normally
    dataset_test_shuffled = SVHN(root='./data', split='test', download=True)
    
    # 2. Convert to float tensor [N, C, H, W] in range [0, 1]
    # Note: SVHN is already [N, C, H, W], so we can drop the permute(0,1,2,3) entirely
    shuffled_data = torch.tensor(dataset_test_shuffled.data).float() / 255.0
    
    # 3. Apply Fixed Pixel Permutation DIRECTLY to the data array
    h, w = 32, 32
    perm_idx = torch.randperm(h * w)
    
    # Flatten spatial dims [N, C, H*W], permute, and reshape back to [N, C, H, W]
    N, C, _, _ = shuffled_data.shape
    
    # FIX: Use .reshape() instead of .view() to handle memory contiguity
    shuffled_data_flat = shuffled_data.reshape(N, C, -1)
    shuffled_data_permuted = shuffled_data_flat[:, :, perm_idx].reshape(N, C, h, w)
    
    # Overwrite data tensor and set default transform to empty
    dataset_test_shuffled.data = shuffled_data_permuted
    dataset_test_shuffled.transform = lambda x: x  # Identity transform
    
    testloader_shuffled = torch.utils.data.DataLoader(
        dataset_test_shuffled, 
        batch_size=batch_size, 
        shuffle=False,
        generator=torch.Generator(device='cuda'), 
        num_workers=0
    )
    
    features_normal = {}
    features_gmu = {}
    handle_norm = resnet18.act1.register_forward_hook(get_features(features_normal, 'conv1'))
    handle_gmu = resnet18_gmu.act1.register_forward_hook(get_gmu_matched_features(features_gmu, 'conv1'))        

    print("\n================ Testing CNN (Normal ResNet18) ================")
    cnn_results = evaluate_activation_shifts(resnet18, testloader, testloader_shuffled, features_normal, 'conv1')

    print("\n================ Testing GMU (ResNet18 GMU) ================")
    gmu_results = evaluate_activation_shifts(resnet18_gmu, testloader, testloader_shuffled, features_gmu, 'conv1')

    # Optional: Plot the bar chart comparing the collapse
    plot_activation_collapse(cnn_results, gmu_results)
    
    cnn_roughness = compute_normalized_spatial_roughness(
        resnet18.conv1.weight, is_gmu=False
    )
    gmu_roughness = compute_normalized_spatial_roughness(
        resnet18_gmu.conv1.weights, is_gmu=True
    )
    
    print(
        f"Normalized Spatial Roughness (Lower = Smoother/Higher Spatial Correlation):"
    )
    print(f"  CNN Conv1 : {cnn_roughness:.4f}")
    print(f"  GMU Conv1 : {gmu_roughness:.4f}")

    plot_and_quantify_spatial_structure(resnet18, resnet18_gmu)
    
    
    # base_x_cnn, _ = extract_first_layer(resnet18, testloader, features_normal, 'conv1')
    # shuf_x_cnn, _ = extract_first_layer(resnet18, testloader_shuffled, features_normal, 'conv1')
    
    # base_x_gmu, _ = extract_first_layer(resnet18_gmu, testloader, features_gmu, 'conv1')
    # shuf_x_gmu, _ = extract_first_layer(resnet18_gmu, testloader_shuffled, features_gmu, 'conv1')
    
    # # 1. Plot histograms
    # plot_activation_distributions(base_x_cnn, shuf_x_cnn, base_x_gmu, shuf_x_gmu)

        
    # Call the visualizer
    # visualize_spatial_weights(resnet18, resnet18_gmu)

  