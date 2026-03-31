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
from models import *
from requisites import *
import torchvision
import seaborn as sns
from corruptions import *


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
    batch_size = 200
    training_size = 60000
    
    dataset,dataset_test,my_dataset,my_dataset_test,trainloader,testloader = load_data_and_generators(dataset_name,training_size,rescale)

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
        
        

    net_normal1 = Net_vanilla_CNN(input_channels,layers,kernels,multiplier, gmu_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)
    
    net_normal2 = Net_vanilla_CNN(input_channels,layers,kernels,multiplier, gmu_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)

    
    
    
    net_gmu = Net_vanilla_CNN_convert_smaller(input_channels,layers,kernels,multiplier,num_slices,gmu_epsilons,decay_regress,decay_errors,use_bn=use_bn,dropping=dropping)

    # net_normal1.load_state_dict(torch.load('./MNIST_Normal_seed1.h5',weights_only=True))
    # net_normal2.load_state_dict(torch.load('./mnist_noaug.h5',weights_only=True))
    # net_gmu.load_state_dict(torch.load('./mnist_gmu_noaug.h5',weights_only=True))
    # net_normal1.load_state_dict(torch.load('./fashionmnist_Normal_seed1.h5',weights_only=True))
    # net_normal2.load_state_dict(torch.load('./fashionmnist_withaug.h5',weights_only=True))
    # net_gmu.load_state_dict(torch.load('./fashionmnist_gmu_withaug.h5',weights_only=True))
    
    net_normal1 = net_normal1.eval()
    net_normal2 = net_normal2.eval()
    net_gmu = net_gmu.eval()
    
    # cka_output = averaged_cka_matrix(net_normal1, net_normal2, (nn.Conv2d,), corrupted_loaders)

    # # Calculate CKA
    
    # model1_layers=["gmu1","conv2","conv3","conv4","bnorm_fc","fc1"]
    # model2_layers=["conv1","conv2","conv3","conv4","bnorm_fc","fc1"]
    
    # cka_output, mean_diag = averaged_cka_matrix(net_gmu, net_normal1, 
    #                         model1_layers, model2_layers, 
    #                         corrupted_loaders, device="cuda")
    
    net_args = dict(
        input_channels=input_channels,
        layers=layers,
        kernels=kernels,
        multiplier=multiplier,
        epsilons=gmu_epsilons,
        decay_regress=decay_regress,
        decay_errors=decay_errors,
        use_bn=use_bn,
        dropping=dropping
    )

    cka_output, diag_mean = averaged_seed_cka_matrix(
        Net_vanilla_CNN,
        net_args,
        model1_layers=["conv1","conv2","conv3","conv4","bnorm_fc","fc1"],
        model2_layers=["conv1","conv2","conv3","conv4","bnorm_fc","fc1"],
        dataloader=testloader,
        seeds=[0,1,2,3,4,5,6,7,8,9,10],
        device="cuda"
    )
    
    print("Mean diagonal similarity across seeds:", diag_mean.item())
    

    
    # cka = CKA(net_normal1, net_normal2,
    #       model1_name="V1",   # good idea to provide names to avoid confusion
    #       model2_name="V2",   
    #       model1_layers=["conv1","conv2","conv3","conv4","bnorm_fc","fc1"], # List of layers to extract features from
    #       model2_layers=["conv1","conv2","conv3","conv4","bnorm_fc","fc1"], # extracts all layer features by default
    #       device='cuda')
    
    # cka.compare(testloader)
    # # # cka_output = cka_output[::2, ::2]
    # results = cka.export()
    
    # # print(torch.mean(results['CKA'].diag()))
    # cka_output = results['CKA']
    
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