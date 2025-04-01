# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:30:11 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:49:23 2024

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
import pandas
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

import sklearn.datasets
from scipy import stats
from scipy.stats import multivariate_normal

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

from make_data import * 
from resnet import *
from sklearn.model_selection import train_test_split
import openml 


from sklearn.base import TransformerMixin


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




import gc
import itertools as it 

    

from copy import copy 


def train_network_normal(net,trainloader, init_rate,epochs,weight_decay,testloader,my_dataset_test):
    net = net
    net = net.cuda()
    net = net.train()
    optimizer = optim.Adam(net.parameters(), lr=init_rate, weight_decay=weight_decay)
# 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15,T_mult=2)
    init_epoch = 0
    all_train_losses = []
    train_loss_min = 9999
    weights = [] 
    g = len(torch.unique(trainloader.dataset.labels))
    for i in range(g):
        weights.append(1/(trainloader.dataset.labels==i).float().mean())
    weights = np.array(weights)/g
    # weights = weights/np.mean(weights)
    weights = torch.from_numpy(np.array(weights)).cuda()
    criterion = nn.CrossEntropyLoss(weight = weights)
    mse_loss = nn.MSELoss()
    net_best = deepcopy(net)
    test_list = [] 
    for epoch in range(epochs):

        #
        train_loss = []
        loss_weights = [] 
       
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # inputs = inputs.cuda()
            # labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            
            allouts,x_errs = net(inputs)
            if hasattr(net, 'gmu1'):
              
                loss = criterion(allouts, labels.long()) 
#               
            else:
                loss = criterion(allouts, labels.long())
                            #+0.1*(torch.abs(net.conv1.bias).mean())#+ criterion(x_errs, labels.long()) + net.decay_errors*torch.mean(-x_errs)
            loss.backward()
            train_loss.append(loss.item())
            # if epoch>100:
                # print('h')
            loss_weights.append(len(labels))
            
            # print(1)
            # print(torch.mean(1-x_errs))
            optimizer.step()
        # print(time.time()-T)
            # print(0)
            # print('here')
        for g in optimizer.param_groups:
            g['lr'] = init_rate* (0.5 ** (epoch // 100))
        
        all_train_losses.append(np.average(np.array(train_loss),weights=np.array(loss_weights)))
        
    
            
        if all_train_losses[-1] < train_loss_min:
            train_loss_min = copy(all_train_losses[-1])
            net_best = deepcopy(net)
    
    net_best = net_best.eval()
    return net_best,all_train_losses


def scale_dataset(dataset_old,scale):
    if scale == 1.0:
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


NoneType = type(None)
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):
        # print(y)
        if type(y) == NoneType:
            self.fill = pandas.Series([X[c].value_counts().index[0]
                if X[c].dtype == 'category' else X[c].median() for c in X],
                index=X.columns)
        else:
            self.fill = y

        return self

    def transform(self, X, y=None):
        # print(self.fill)
        return X.fillna(self.fill),self.fill


def dataframe_to_torch(X,y):
    
    X = pandas.get_dummies(X)
    
    Tx = pandas.DataFrame.to_numpy(X)
    TX = np.zeros(Tx.shape)
    Ty,cats = pandas.factorize(y,sort=True)
    for i in range(Tx.shape[1]):
        if X.dtypes[i].name == 'category':
            # print('halle')
            Tx[:,i],cats = pandas.factorize(Tx[:,i],sort=True)
            TX[:,i] = Tx[:,i].astype('float32')
        else:
            TX[:,i] = Tx[:,i].astype('float32')
    
    # y = 
    X = torch.from_numpy(TX).float()
    y = torch.from_numpy(Ty)
    return X,y

def  load_data_and_generators(dataset_name,partition,conversion,labelnoise):
    
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

    task = openml.tasks.get_task(task_id=int(dataset_name), download_splits=False,download_data=False,download_qualities=False,download_features_meta_data=False)
    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute,
    )
    
    
    if isinstance(y[1], bool):
        y = y.astype('bool')
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=1-partition,
        random_state=11,
        stratify=y,
        shuffle=True,
    )
    train_column_nan_info = X_train.isna().all()
    test_column_nan_info = X_test.isna().all()
    only_nan_columns = [label for label, value in train_column_nan_info.items() if value]
    test_nan_columns = [label for label, value in test_column_nan_info.items() if value]
    only_nan_columns.extend(test_nan_columns)
    only_nan_columns = set(only_nan_columns)
    X_train.drop(only_nan_columns, axis='columns', inplace=True)
    X_test.drop(only_nan_columns, axis='columns', inplace=True)
    
    # X = X.cuda()  #train_dataset.train_data is a Tensor(input data)
    # y = y.cuda()
    X_train,fill = DataFrameImputer().fit_transform(X_train)
    X_test,__ = DataFrameImputer().fit_transform(X_test,y=fill)
    
    X_train,y_train = dataframe_to_torch(X_train, y_train)
    X_test,y_test = dataframe_to_torch(X_test, y_test)
    X_train = X_train.unsqueeze(2).unsqueeze(3).cuda()
    X_test = X_test.unsqueeze(2).unsqueeze(3).cuda()
    # y_test = 1 - y_test
    
    if conversion == 'rank':
        print('R')
        X_train,params = rank_convert_data(X_train)
        X_test = rank_convert_data(X_test,params)
    elif conversion == 'uniform':
        # X = uniform_convert_data(X)
        X_train,params = uniform_convert_data(X_train)
        X_test = uniform_convert_data(X_test,params)
    elif conversion == 'normalize':
        # X = normalized_convert_data(X)
        X_train,params = normalized_convert_data(X_train)
        X_test = normalized_convert_data(X_test,params)
    
    medians,indstemp = torch.median(X_train,dim=0)
    my_dataset = Dataset(dataset_name, X_train, y_train)
    my_dataset_test = Dataset(dataset_name, X_test, y_test)

    trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                          shuffle=True,generator=torch.Generator(device='cuda'), num_workers=0)
    testloader = torch.utils.data.DataLoader(my_dataset_test, batch_size=batch_size,
                                          shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)
    
    # testloader_adversary = torch.utils.data.DataLoader(my_dataset_test, batch_size=1,
    #                                       shuffle=False,generator=torch.Generator(device='cuda'), num_workers=0)

    
    return  my_dataset,my_dataset_test,trainloader,testloader,medians,dataset.name


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


def rank_convert_data(data):
    
    for i in range(data.shape[1]):
        temp,data[:,i,0,0] = torch.unique(data[:,i,0,0],return_inverse=True)
        data[:,i,0,0]  = data[:,i,0,0]/torch.max(data[:,i,0,0]) 
        # dataset.data = dataset.data
    
    return data

def uniform_convert_data(data,params=None):
    if params == None:
        params = [] 
        for i in range(data.shape[1]):
            params.append([torch.min(data[:,i,0,0]),torch.max(data[:,i,0,0])])
            if torch.max(data[:,i,0,0])>torch.min(data[:,i,0,0]):
                data[:,i,0,0] = (data[:,i,0,0] - torch.min(data[:,i,0,0])) / (torch.max(data[:,i,0,0]) - torch.min(data[:,i,0,0])) 
            # else:
            #     print('mambaaaaaa')
            # dataset.data = dataset.data
        return data,params
        
    else:
        for i in range(data.shape[1]):
            if params[i][1]>params[i][0]:
                data[:,i,0,0] = (data[:,i,0,0] - params[i][0]) / (params[i][1] - params[i][0]) 
        return data


def normalized_convert_data(data,params=None):
    if params == None: 
        params = [] 
        for i in range(data.shape[1]):
            params.append([torch.mean(data[:,i,0,0]),torch.std(data[:,i,0,0])])
            if torch.std(data[:,i,0,0])>0:
                data[:,i,0,0] = (data[:,i,0,0] - torch.mean(data[:,i,0,0])) / (torch.std(data[:,i,0,0]))
            
        return data, params
    else:
        for i in range(data.shape[1]):
            if params[i][1]>0:
                data[:,i,0,0] = (data[:,i,0,0] - params[i][0]) /  (params[i][1])
        return data

                                                                       


def test_network(net, testloader, test_labels,input_noise=0):
    net = net.eval()

    accuracy = torch.tensor(0)
    dataiter = iter(testloader)
    weights = [] 
    g = len(torch.unique(testloader.dataset.labels))
    for i in range(g):
        weights.append(1/((testloader.dataset.labels==i).float().sum()*g))
                       
    weights = np.array(weights)
    weights = torch.from_numpy(weights).cuda() 
    # total = 0 
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs  
            all_outs,temp = net(inputs)
            predicted = torch.argmax(all_outs,1)
            accuracy = accuracy + torch.sum((predicted == labels).float()*(weights[labels]))
    return accuracy


# +
if __name__ == "__main__":
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')
    torch.backends.cudnn.deterministic = True
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    torch.cuda.manual_seed(0)
    
    atanh_convert = 0 

    gc.collect()
    torch.cuda.empty_cache()
    results = [] 
    test_srn = []
    test_normal  = []
    train_srn = [] 
    train_normal = [] 
    trials = 1
    names = [233088,233090,233091,	233093,	233094,	233096,	233103,	233108,	233109,	233115,	233116,	233117,		233134,	233135,		233099,	233107,	233110,	233119,	233124,	233122,	233112,	233114,	233120,	233102,	233106,	233142,	233104]
    for d in range(len(names)):
        for i in range(trials):
            
            dataset_name = str(names[d])
            # dataset_name = 233092
            batch_size = 200
            init_rate = 0.001
#             init_rate = 0.01
            
            init_rate_crank = 0.01
            labelnoise = 0
            input_noise = 0 
            partition = 0.8
            
            epsilons = [0.0000001 ,0.0001,0.0001]
            num_slices = 0
            degree = 1
            normalize = True
            use_dropout = True
            
            
            
            step_size = 10
            gamma_learning = 0.8
            total_epoch = 200
            total_epoch_crank = 200
            decay_normal = 0   
            decay_errors = 0
            decay_regress = 0
            dropping = 0          
            lower = 1.0
            upper = 1.0
            
            # decay_normal_crank = 0 
            
            layers = [200]
            layers_crank = []
            
            training_size = 10000
            mode = 'regress_batch'
            use_bn = True
            conversion = 'uniform' #Ensures that the input dimensions are between 0 and 1 (Same as Kadra et al. (2021)). 
            global DEGREE
            DEGREE = 4
            alpha= 0.2
            
        
            my_dataset,my_dataset_test,trainloader,testloader,medians,data_name = load_data_and_generators(dataset_name,partition,conversion,labelnoise)
            classes = len(torch.unique(my_dataset.labels))
            print(classes)
            print(my_dataset.inputs.shape)
            input_channels = my_dataset.inputs.shape[1]
            net = ResGMUMLP(input_channels, BasicBlock, [1,1],medians,num_slices=num_slices, degree=degree, normalize=normalize, epsilon = epsilons[0],num_classes = classes,use_dropout=use_dropout)
            net.mode = mode
            
            net,all_losses = train_network_normal(net,trainloader, init_rate,total_epoch,decay_normal,testloader,my_dataset_test)
            train_accuracy = test_network(net, trainloader, my_dataset.labels,0)
            accuracy_srn = test_network(net, testloader, my_dataset_test.labels, 0)
            print(accuracy_srn)
            results.append({
            "Dataset": data_name,
            "Trial": i + 1,
            "Train Accuracy": train_accuracy,
            "Test Accuracy GMU": accuracy_srn
            })
# -
    output_file = "accuracies.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results written to {output_file}")


