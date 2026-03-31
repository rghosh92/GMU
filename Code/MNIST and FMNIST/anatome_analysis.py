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
# from cka import CKACalculator
from models import *
from requisites import *
import torchvision
import seaborn as sns
from anatome import Distance

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
                                          shuffle=True,generator=torch.Generator(device='cuda'), num_workers=0)
    testloader = torch.utils.data.DataLoader(my_dataset_test, batch_size=batch_size,
                                          shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)
    
    

    
    return dataset,dataset_test, my_dataset,my_dataset_test,trainloader,testloader


from anatome import CCAHook

def get_layer_distance(model1, model2, layer1, layer2, dataloader, size=8, device="cuda"):
  
    model1.eval().to(device)
    model2.eval().to(device)

    hook1 = CCAHook(model1, layer1)
    hook2 = CCAHook(model2, layer2)

    # Run one batch through both models
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            model1(x)
            model2(x)
            break  # just one batch is enough for similarity

    score = hook1.distance(hook2, size=size)

    # Cleanup
    hook1.clear()
    hook2.clear()
    torch.cuda.empty_cache()

    return score



if __name__ == "__main__":
    
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    
    dataset_name = "MNIST"
    num_slices = 3

    
    dataset,dataset_test,my_dataset,my_dataset_test,trainloader,testloader = load_data_and_generators(dataset_name,training_size,rescale)

    net_normal1 = Net_vanilla_CNN(input_channels,layers,kernels,multiplier, gmu_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)
    net_normal2 = Net_vanilla_CNN(input_channels,layers,kernels,multiplier, gmu_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)

    
    
    net_gmu = Net_vanilla_CNN_convert_smaller(input_channels,layers,kernels,multiplier,num_slices,gmu_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)

    net_normal1.load_state_dict(torch.load('./MNIST_Normal_seed1.h5',weights_only=True))
    net_normal2.load_state_dict(torch.load('./mnist_noaug.h5',weights_only=True))
    net_gmu.load_state_dict(torch.load('./mnist_gmu_noaug.h5',weights_only=True))
    
    layer_names = ["conv1", "conv2", "conv3", "conv4"]    
    
    cka_output = get_layer_distance(net_normal1, net_normal2,"conv3","conv3", testloader)
    
    # Plot with fixed scale
    
    plt.rcParams['figure.figsize'] = (8, 8)
    
    sns.heatmap(
        cka_output.cpu().numpy(),
        cmap="inferno",
        vmin=0, vmax=1,        # fix scale to [0,1]
        square=True,
        cbar_kws={"label": "CKA similarity"},
        annot=True,            # show numbers in each cell
        fmt=".2f",             # format to 2 decimal places
        annot_kws={"size": 8}  # control font size of annotations
    )
    
    plt.title("CKA Similarity Matrix (Conv2d layers)")
    plt.xlabel("Model 2 layers")
    plt.ylabel("Model 1 layers")
    plt.tight_layout()
    plt.show()