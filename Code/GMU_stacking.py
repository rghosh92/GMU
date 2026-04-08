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


    
class SimpleGMULayer(nn.Module):
    def __init__(self,input_channels, output_channels, epsilon = 0.0001, num_slices=2,degree=4,exponent=True, normalize = True,Type = 'exp'):
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
        self.Type = Type
        
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
            if self.Type != 'exp':
                y = y - (self.weight_bias.unsqueeze(0).repeat(y.shape[0],1,1))
            
            if self.num_slices == 0:
                err = torch.mean((y)**2,dim=2)
                # err = err.unsqueeze(2).unsqueeze(3)
                return -err
                
            
            if self.normalize == True and self.Type == 'exp':
                y = y/(torch.std(y,2)).unsqueeze(2).repeat(1,1,self.weights.shape[1])
            
            X = self.weights
            
            if self.Type == 'exp':
                X = torch.concat((torch.ones((X.shape[0],X.shape[1],1)), X),dim=2)

            
            X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
            X_cov_inv = torch.linalg.inv(X_cov)
            M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
            
            
            W = torch.einsum('ijk,bik->ijb',M,y)
            
            pred_final = torch.einsum('bij,bjk->bik', X, W)
            
            pred_final = pred_final.permute(2,0,1)
            
            err = torch.mean((y-pred_final)**2,dim=2)
            # err = err.unsqueeze(2).unsqueeze(3)
            if self.Type == 'exp':
                return torch.exp(-err)
            else:
                return -err
        
        

class DeeperGMUMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                  num_slices,epsilon=1e-4, normalize=True, use_dropout=True):
        super(DeeperGMUMLP, self).__init__()
        
        # Single hidden layer
        self.gmu_in = SimpleGMULayer(input_dim, hidden_dim,
                                      epsilon=epsilon,  num_slices=num_slices[0], normalize=normalize)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        # self.drop = nn.Dropout(0.2) if use_dropout else nn.Identity()
        
        # Final GMU layer as classifier
        self.gmu_out = SimpleGMULayer(hidden_dim, output_dim,
                                      epsilon=epsilon,num_slices=num_slices[1], normalize=False,Type='naturale')

    def forward(self, x):
        # Hidden layer
        out = self.gmu_in(x)
        out = F.relu(self.bn1(out))
        
        # Dropout
        # out = self.drop(out)
        
        # Final GMU output
        out = self.gmu_out(out)
        
        return out


class GMUMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 epsilon=1e-4, num_slices=[5,2], normalize=True, use_dropout=True):
        super(GMUMLP, self).__init__()
        
        # Hidden GMU layer
        self.gmu_in = SimpleGMULayer(input_dim, hidden_dim,
                                     epsilon=epsilon, num_slices=num_slices[0], normalize=normalize)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        # self.drop = nn.Dropout(0.2) if use_dropout else nn.Identity()
        
        # Final fully connected classifier
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Hidden GMU + BN
        out = self.gmu_in(x)
        out = F.relu(self.bn1(out))
        
        # Dropout
        # out = self.drop(out)
        
        # Final linear output (logits)
        out = self.fc_out(out)
        
        return out






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
            
            allouts = net(inputs)
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
       
    # print(all_train_losses[-1])

        
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


def load_data_and_generators(dataset_name, training_size, scale,batch_size=64):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(
            root='./../../../data', train=True, download=True, transform=transform)
        dataset_test = torchvision.datasets.FashionMNIST(
            root='./../../../data', train=False, download=True, transform=transform)
        
    elif dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(
            root='./../../../data', train=True, download=True, transform=transform)
        dataset_test = torchvision.datasets.MNIST(
            root='./../../../data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Apply scaling if you have a custom function
    dataset = scale_dataset(dataset, scale)
    dataset_test = scale_dataset(dataset_test, scale)
    
    # Flatten to 1D vectors: (N, 28*28)
    dataset.data = dataset.data.float().view(dataset.data.size(0), -1)
    dataset_test.data = dataset_test.data.float().view(dataset_test.data.size(0), -1)
    
    # Restrict training size
    dataset.data = dataset.data[:training_size]
    dataset.targets = dataset.targets[:training_size]
    
    # Move to GPU
    dataset.data = dataset.data.cuda()
    dataset.targets = dataset.targets.cuda()
    dataset_test.data = dataset_test.data.cuda()
    dataset_test.targets = dataset_test.targets.cuda()
    
    # Wrap in your custom Dataset class
    my_dataset = Dataset(dataset_name, dataset.data, dataset.targets)
    my_dataset_test = Dataset(dataset_name, dataset_test.data, dataset_test.targets)
    
    trainloader = torch.utils.data.DataLoader(
        my_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator(device='cuda'), num_workers=0)
    
    testloader = torch.utils.data.DataLoader(
        my_dataset_test, batch_size=batch_size, shuffle=False,
        generator=torch.Generator(device='cuda'), num_workers=0)
    
    return dataset, dataset_test, my_dataset, my_dataset_test, trainloader, testloader



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
            inputs = inputs  
            all_outs = net(inputs)
            predicted = torch.argmax(all_outs,1)
            correct = correct + torch.sum(predicted == labels)
    accuracy = float(correct) / float(len(test_labels))
    return accuracy

# def test_network_corrupted(net, dataset_name, corruptions,atan_convert):
#     net = net.eval()
#     accuracy_list = [] 
#     for corruption in corruptions:
#         data_test = np.load('./'+ dataset_name+ '_c/'+ corruption +'/test_images.npy')
#         labels_test = np.load('./'+ dataset_name+ '_c/'+ corruption +'/test_labels.npy')
#         # data_test = (torch.from_numpy(data_test)).view(data_test.shape[0],
#                                                      # int(data_test.size/data_test.shape[0]),1,1)
#         labels_test = torch.from_numpy(labels_test)
#         data_test = (torch.from_numpy(data_test)).float().squeeze().unsqueeze(1)/255.0
#         # if atan_convert>0:
#         #     data_test = map_inf(data_test, atan_convert)
        
#         my_dataset_test = Dataset(dataset_name, data_test.cuda(), labels_test.cuda())

#         testloader = torch.utils.data.DataLoader(my_dataset_test, batch_size=batch_size,
#                                               shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)
        
        
#         correct = torch.tensor(0)
#         dataiter = iter(testloader)
#         with torch.no_grad():
#             for i, data in enumerate(testloader, 0):
#                 # get the inputs
#                 inputs, labels = data
#                 all_outs,temp = net(inputs)
#                 predicted = torch.argmax(all_outs,1)

#                 correct = correct + torch.sum(predicted == labels)
#         accuracy = float(correct) / float(len(labels_test))
#         accuracy_list.append(accuracy)
#         print("Corruption:",corruption, " Accuracy: ", accuracy)
    
#     print("Mean accuracy:",np.mean(accuracy_list))
#     return 0





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




def generate_corrupted_tensors(base_dataset, corruption_fn, img_size=(28, 28)):
    corrupted_imgs = []
    corrupted_labels = []
    
    for i in range(len(base_dataset)):
        img, label = base_dataset[i]  # img is 1D vector now
        
        # Reshape to 2D for corruption
        img_2d = img.view(*img_size).cpu().numpy()
        
        # Apply corruption
        corrupted_img = corruption_fn(255 * img_2d)
        
        # Convert back to tensor and flatten to 1D
        corrupted_img = torch.from_numpy(corrupted_img).float().view(-1)
        
        corrupted_imgs.append(corrupted_img)
        corrupted_labels.append(label)
    
    # Stack into [N, D] and move to GPU
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
                all_outs = net(inputs)
                predicted = torch.argmax(all_outs, 1)
                correct += (predicted == labels).sum().item()
            accuracy = correct / len(loader.dataset)
            accuracy_list.append(accuracy)
            # print(f"Corruption: {name}, Accuracy: {accuracy:.4f}")
    # print("Mean accuracy:", np.mean(accuracy_list))
    return np.mean(accuracy_list)



import seaborn as sns


 
    
    
    # a = input("")
        
        


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

    gc.collect()
    torch.cuda.empty_cache()
    
    dataset_name = "FashionMNIST"
    
    init_rate = 0.0005
    init_rate_crank = 0.01
    labelnoise = 0
    input_noise = 0 
    input_noise_array = [0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.25,0.3,0.35]
    epsilons = [ .001]
    gmu_epsilons = [0.0001,0.00001,0]
    
    # test_transforms = ['scale','translate','rotate','']
    
    
    total_epoch_crank = 100
    decay_normal = 0    
    dropping = 0 
    rescale = 1.0
    training_size_list = [5000]
    training_size = 1000
    input_channels = 784
    hidden_dim = 200
    output_dim = 10
    batch_size = 200
    total_epoch = 200

    
    # ablation_slices = [1,2,3,4,5,6,7,8]
    # average_corruption_accuracies = [] 
    normal_accuracies = [] 
    
    for training_size in training_size_list:
        
        print("Training size:", training_size)
        
        dataset,dataset_test,my_dataset,my_dataset_test,trainloader,testloader = load_data_and_generators(dataset_name,training_size,rescale)
       
        acc_list = [] 
        corracc_list = [] 
        
        for temp in range(10):   
           
            net = DeeperGMUMLP(input_channels, hidden_dim, output_dim, num_slices =[5,3],epsilon=gmu_epsilons[0])
            
            net,all_losses = train_network_normal(net,trainloader,testloader,my_dataset_test.labels, init_rate,total_epoch,decay_normal)
            
            
            acc =  test_network(net, trainloader, my_dataset.labels, 0)
            # print('Train Accuracy Deep-GMU:', acc)
            
            acc =  test_network(net, testloader, my_dataset_test.labels, 0)
            # print('Test Accuracy Deep-GMU:', acc)
            acc_list.append(acc)
            
            accor = test_network_corrupted_loaders(net, corrupted_loaders)
            # print('Test Accuracy Deep-GMU on Corruptions:', accor)
            corracc_list.append(accor)
        print('GMU-2:')
        
        print(np.mean(acc_list))
        print(np.std(acc_list))
        print(np.mean(corracc_list))
        print(np.std(corracc_list))
        
        acc_list = [] 
        corracc_list = [] 
        
        for temp in range(10):   
            
            net = GMUMLP(input_channels, hidden_dim, output_dim, num_slices =[5,1],epsilon=gmu_epsilons[0])
            
            net,all_losses = train_network_normal(net,trainloader,testloader,my_dataset_test.labels, init_rate,total_epoch,decay_normal)
            
            
            acc =  test_network(net, testloader, my_dataset_test.labels, 0)
            # print('Test Accuracy GMU-MLP:', acc)
            acc_list.append(acc)
    
            
            accor = test_network_corrupted_loaders(net, corrupted_loaders)
            # print('Test Accuracy GMU-MLP on Corruptions:', accor)
            corracc_list.append(accor)
        print('GMU-MLP:')
    
        print(np.mean(acc_list))
        print(np.std(acc_list))
        print(np.mean(corracc_list))
        print(np.std(corracc_list))
     
     
     
    # net = SimpleMLP(input_channels, hidden_dim, output_dim)
    # net,all_losses = train_network_normal(net,trainloader,testloader,my_dataset_test.labels, init_rate,total_epoch,decay_normal)
    
    # acc =  test_network(net, testloader, my_dataset_test.labels, 0)
    # print('Test Accuracy MLP:', acc)
    
        
        # accor = test_network_corruptions(net, testloader, corruptions,atanh_convert)
        # average_corruption_accuracies.append(accor)
    
        
        # test_network_corruptions_with_bn_adaptation(net, testloader, corruptions)    


    
    
