# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:33:34 2026

@author: User
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
# from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import resnet18
from tqdm.autonotebook import tqdm
from copy import deepcopy
from torch import nn
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils import data
import random
from torch_cka import CKA
import torchvision
import seaborn as sns
from corruptions import *
import gc
from data_loaders import *
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
    

class Shallow_GMUCNN(nn.Module):
    def __init__(self,input_channels,layers,kernels,epsilons = 0.001, classes = 5,use_bn = True,dropping=0,poly_order_init=5):
        super(Shallow_GMUCNN, self).__init__()
        
        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons 
        self.use_bn = True
        # network layers
        self.convs = []
        self.bns = []
        self.Inds_weight = [] 
        self.Mul_mats = [0] 
        self.bns_rank = [] 
        self.convs = [] 
        self.dropping = 0 
        # print(kernels[0])
#         self.srn1 = SRNLayer(input_channels, 64, kernels[0],padding=int((kernels[0]-1)/2),epsilon = epsilons[0],num_slices=3,degree=1,normalize=True)
        self.srn1 = SRNLayer(input_channels, 64, kernels[0],padding=0,epsilon = epsilons,num_slices=1,degree=1)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=0)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=0)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=0)

        self.mpool1 = nn.AvgPool2d(2)
        self.mpool2 = nn.AvgPool2d(2)
        self.mpool3 = nn.MaxPool2d(2)
        self.mpool4 = nn.MaxPool2d(4)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)


        self.bnorm_fc = nn.BatchNorm2d(128)
        self.fc1 = nn.Conv2d(128,128,1)
        self.fc2 = nn.Conv2d(128,classes,1)

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
       
        
    def forward(self, x,bazinga=1):
        x = self.srn1(x)
        x = self.feat_net(x)
        x_errs = x
        
        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
#         xm = self.drop(xm)
        
#         xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)
        
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm
    
    
class Shallow_CNN(nn.Module):
    def __init__(self,input_channels,layers,kernels,epsilons = 0.001, classes = 5,use_bn = True,dropping=0,poly_order_init=5):
        super(Shallow_CNN, self).__init__()
        
        self.layers = layers
        self.post_filter = False
        self.epsilons = epsilons 
        self.use_bn = True
        # network layers
        self.convs = []
        self.bns = []
        self.Inds_weight = [] 
        self.Mul_mats = [0] 
        self.bns_rank = []  
        self.convs = [] 
        self.dropping = dropping
        # print(kernels[0])
#         self.srn1 = SRNLayer(input_channels, 64, kernels[0],padding=int((kernels[0]-1)/2),epsilon = epsilons[0],num_slices=3,degree=1,normalize=True)
#         self.srn1 = SRNLayer(input_channels, 64, kernels[0],padding=int((kernels[0]-1)/2),epsilon = epsilons,num_slices=3,degree=1)
        
        self.conv1 = nn.Conv2d(input_channels,64,5, padding=0)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=0)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=0)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=0)

        self.mpool1 = nn.MaxPool2d(2)
        self.mpool2 = nn.MaxPool2d(2)
        self.mpool3 = nn.MaxPool2d(2)
        self.mpool4 = nn.MaxPool2d(4)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)


        self.bnorm_fc = nn.BatchNorm2d(128)
        self.fc1 = nn.Conv2d(128,128,1)
        self.fc2 = nn.Conv2d(128,classes,1)

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
        x = self.feat_net(x)
        x_errs = x

        
        xm = x.view(
            [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        
        
#         xm = self.relu(self.bnorm_fc(self.fc1(xm)))
        xm = self.fc2(xm)
        
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm
    

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


def  load_data_and_generators(dataset_name,training_size,scale):
    
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
                                          shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)
    testloader = torch.utils.data.DataLoader(my_dataset_test, batch_size=batch_size,
                                          shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)
    
    

    
    return dataset,dataset_test, my_dataset,my_dataset_test,trainloader,testloader


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
    



def averaged_cka_matrix(model1, model2, 
                        model1_layers, model2_layers, 
                        dataloaders, device="cuda"):
   
    cka = CKA(model1, model2,
              model1_name="Model1",
              model2_name="Model2",
              model1_layers=model1_layers,
              model2_layers=model2_layers,
              device=device)
    
    matrices = []
    for name, loader in dataloaders.items():
        cka.compare(loader)
        results = cka.export()
        matrices.append(results['CKA'])
    
    # Stack and average
    stacked = torch.stack(matrices)
    averaged_matrix = stacked.mean(dim=0)
    
    mean_diag = torch.mean(torch.diag(averaged_matrix))
    
    return averaged_matrix, mean_diag


def averaged_seed_cka_matrix(
    NetClass,                # your network class (e.g. Net_vanilla_CNN)
    net_args,                # dict of arguments to initialize NetClass
    model1_layers, model2_layers,
    dataloader,
    seeds=[0,1,2,3,4],       # list of seeds to average over
    device="cuda"
):
   
    matrices = []
    
    for seed in seeds:
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Initialize two networks with same seed
        net_normal1 = NetClass(**net_args)
        net_normal2 = NetClass(**net_args)
        
        # Build CKA object
        cka = CKA(net_normal1, net_normal2,
                  model1_name="Net1",
                  model2_name="Net2",
                  model1_layers=model1_layers,
                  model2_layers=model2_layers,
                  device=device)
        
        # Compare on dataloader
        cka.compare(dataloader)
        results = cka.export()
        matrices.append(results['CKA'])
    
    # Average across seeds
    stacked = torch.stack(matrices)
    averaged_matrix = stacked.mean(dim=0)
    mean_diag = torch.mean(torch.diag(averaged_matrix))
    
    return averaged_matrix, mean_diag


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
        # test_loader_norb = load_norb(batch_size)
        return train_loader, test_loader

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

import time

if __name__ == "__main__":
    
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    
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
    
    
    trainloader,testloader = load_data_and_generators(dataset_name,training_size,rescale,rank_convert,atanh_convert,labelnoise)
    
    net_gmu = Shallow_GMUCNN(input_channels,layers,kernels,srn_epsilons,use_bn=use_bn,dropping=dropping)

    net_gmu.load_state_dict(torch.load('./stereo_Shallow_GMUCNN_norb_1slices_1degree_5kernel_avgpool_64input_nopadding_moremoreepochs.h5', weights_only=True))

    accuracy = test_network_pairs(net_gmu, testloader, 0)
    
    print('Test Accuracy GMU-CNN (Normal):', accuracy)
        
    
    net_normal = Shallow_CNN(input_channels,layers,kernels,srn_epsilons,use_bn=use_bn,dropping=0)
    net_normal.load_state_dict(torch.load('./stereo_Shallow_CNN_norb_64dim_nopadding_5kernel.h5', weights_only=True))
    
    
    net_normal = net_normal.eval()
    net_gmu = net_gmu.eval()
    
    
    accuracy = test_network_pairs(net_normal, testloader, 0)

    print('Test Accuracy CNN (Normal):', accuracy)
    
    
    cka = CKA(net_gmu, net_normal,
          model1_name="V1",   # good idea to provide names to avoid confusion
          model2_name="V2",   
          model1_layers=["conv1","conv2","conv3","conv4","bnorm_fc","fc1"], # List of layers to extract features from
          model2_layers=["conv1","conv2","conv3","conv4","bnorm_fc","fc1"], # extracts all layer features by default
          device='cuda')
    
    
    # cka_output = averaged_cka_matrix(net_normal1, net_normal2, (nn.Conv2d,), corrupted_loaders)

    # # Calculate CKA
    
    # model1_layers=["gmu1","conv2","conv3","conv4","bnorm_fc","fc1"]
    # model2_layers=["conv1","conv2","conv3","conv4","bnorm_fc","fc1"]
    
    # cka_output, mean_diag = averaged_cka_matrix(net_gmu, net_normal1, 
    #                         model1_layers, model2_layers, 
    #                         corrupted_loaders, device="cuda")
    
    # net_args = dict(
    #     input_channels=input_channels,
    #     layers=layers,
    #     kernels=kernels,
    #     multiplier=multiplier,
    #     epsilons=gmu_epsilons,
    #     decay_regress=decay_regress,
    #     decay_errors=decay_errors,
    #     use_bn=use_bn,
    #     dropping=dropping
    # )

    # cka_output, diag_mean = averaged_seed_cka_matrix(
    #     Net_vanilla_CNN,
    #     net_args,
    #     model1_layers=["conv1","conv2","conv3","conv4","bnorm_fc","fc1"],
    #     model2_layers=["conv1","conv2","conv3","conv4","bnorm_fc","fc1"],
    #     dataloader=testloader,
    #     seeds=[0,1,2,3,4,5,6,7,8,9,10],
    #     device="cuda"
    # )
    
    # print("Mean diagonal similarity across seeds:", diag_mean.item())
    

    
    
    
    # cka.compare(testloader)
    # # # cka_output = cka_output[::2, ::2]
    # results = cka.export()
    
    # # print(torch.mean(results['CKA'].diag()))
    # cka_output = results['CKA']
    
    # Plot with fixed scale
    
    # plt.rcParams['figure.figsize'] = (8, 8)
    
    # sns.heatmap(
    #     cka_output.cpu().numpy(),
    #     cmap="inferno",
    #     vmin=0, vmax=1,        # fix scale to [0,1]
    #     square=True,
    #     cbar_kws={"label": "CKA similarity"},
    #     annot=True,            # show numbers in each cell
    #     fmt=".2f",             # format to 2 decimal places
    #     annot_kws={"size": 8}  # control font size of annotations
    # )
    
    # plt.title("CKA Similarity Matrix (Conv2d layers)")
    # plt.xlabel("Model 2 layers")
    # plt.ylabel("Model 1 layers")
    # plt.tight_layout()
    # plt.show()