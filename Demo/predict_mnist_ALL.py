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
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import shutil

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

from matplotlib.widgets import Button
    




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
        # with torch.no_grad():
        # print(weights[0][0])
        # print(padding)
        layer_out = SRModule_Conv(weights,inputs,p,padding, self.decay_regress,epsilon,  normalize=normalize,exponent=exponent)
        return layer_out
        
    
        
    def forward(self, x,bazinga=1):
       
        if self.mode == 'normal':
            
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
    
    
def predict_sample(net, inputs,samples=20,mode = 'max',min_scale=0.6):
    net = net.eval()
    inputs = inputs.repeat(samples,1,1,1)
    correct = torch.tensor(0)
    
    with torch.no_grad():
        aug = kornia.augmentation.RandomResizedCrop((28,28), scale=(min_scale, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                                                    resample="BILINEAR", same_on_batch=False)
        probs = nn.Softmax(dim=1)
        
        inputs = aug(inputs)
            # inputs = inputs + input_noise*torch.randn_like(inputs)
        
        all_outs,temp = net(inputs)
        predicted = torch.argmax(all_outs,1)
        all_probs = probs(all_outs)
        # print(predicted)
        # print(all_probs.topk(1,dim=1).values.permute(1,0))
        if mode == 'max':
            index = torch.argmax(all_probs.topk(1,dim=1).values)
            return predicted[index],0
        elif mode == 'weighted':
            belief =torch.mean(all_probs,0)
            predicted = torch.argmax(belief)
            return predicted,belief
        elif mode == 'voting':
            vals = torch.mode(predicted).values
            return vals,0




import cv2



# Define input and output directories

import tkinter as tk
from tkinter import messagebox

# Function to display the popup message
def show_popup():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    messagebox.showinfo("Task Complete", 
                        "All images in /Images/MNIST analyzed!\n"
                        "Results stored to /Images/Results-CNN, /Images/Results-CNN_I, and /Images/Results-GMUCNN!")
    root.destroy()  # Close the tkinter instance

# Call the popup function

    

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
    
   
    atanh_convert = 0 

    gc.collect()
    
    torch.cuda.empty_cache()
    
    dataset_name = "MNIST"
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
    
    label_dict = {0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"}
    
    
    # Define variables
    text1 = None
    text2 = None
    text3 = None
    
    dataset_name = "MNIST"
    batch_size = 200
    init_rate = 0.0005
    init_rate_crank = 0.01
    labelnoise = 0
    input_noise = 0
    input_noise_array = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35]
    epsilons = [0.05]
    srn_epsilons = [0.01, 0.00001, 0]
    
    step_size = 10
    gamma_learning = 0.8
    total_epoch = 400
    total_epoch_crank = 100
    decay_normal = 0
    decay_regress = 0
    decay_errors = 0
    dropping = 0
    
    decay_normal_crank = 0
    
    layers = [25, 50]
    kernels = [5, 3]
    layers_crank = []
    
    training_size = 60000
    mode = 'regress_batch'
    use_bn = True
    rescale = 1.0
    rank_convert = False
    input_channels = 1
    
    # Global variable for degrees
    DEGREES = [1, 8]
    alpha = 0.2
    
    # Instantiate and initialize networks
    net2 = Net_vanilla_CNN_convert3(input_channels, layers, kernels, srn_epsilons,
                                    decay_regress, decay_errors, use_bn=use_bn, dropping=dropping)
    net = Net_vanilla_CNN_normal(input_channels, layers, kernels, srn_epsilons,
                                  decay_regress, decay_errors, use_bn=use_bn, dropping=dropping)
    
    # Load pre-trained weights
    net.load_state_dict(torch.load('CNN_normal_MNIST_[3, 8]degree_200epochs_[25, 50]layers60000data.h5', weights_only=True))
    net2.load_state_dict(torch.load('CNN_convert_3slice_1degree_MNIST_[3, 8]degree_300epochs_[25, 50]layers60000data.h5', weights_only=True))
    
    # Set networks to evaluation mode
    net = net.eval()
    net2 = net2.eval()

    # Define input and output directories
  
# Define input and output directories
 
# Define input and output directories

    input_dir = './images/MNIST'
    output_dir_cnn = './images/Results-CNN'
    output_dir_cnn_i = './images/Results-CNN_I'
    output_dir_gmucnn = './images/Results-GMUCNN'
    

    # Remove existing directories if they exist
    for folder in [output_dir_cnn, output_dir_cnn_i, output_dir_gmucnn]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            
    # Create result directories if they don't exist
    os.makedirs(output_dir_cnn, exist_ok=True)
    os.makedirs(output_dir_cnn_i, exist_ok=True)
    os.makedirs(output_dir_gmucnn, exist_ok=True)
    
    # Loop through each image in the input directory
    for image_name in os.listdir(input_dir):
        # Ensure it's a valid image file
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_name)
            
            # Load the original image for saving later
            original_image = Image.open(image_path)
            
            # Preprocess the image for predictions
            Im = original_image.convert('L').resize((28, 28), Image.BILINEAR)
            Im_tensor = torch.from_numpy(np.asarray(Im)).float() / 255.0
    
            # Predict using CNN, CNN_I (inverted inputs), and GMUCNN models
            out_cnn, _ = predict_sample(net, Im_tensor.cuda(), 30, 'voting', min_scale=0.8)
            Im_inverted = 1 - Im_tensor  # Invert the image for CNN_I
            out_cnn_i, _ = predict_sample(net, Im_inverted.cuda(), 30, 'voting', min_scale=0.8)
            out_gmucnn, _ = predict_sample(net2, Im_tensor.cuda(), 30, 'weighted', min_scale=0.8)
    
            # Get predicted classes
            cnn_class = str(out_cnn.cpu().numpy())
            cnn_i_class = str(out_cnn_i.cpu().numpy())
            gmucnn_class = str(out_gmucnn.cpu().numpy())
    
            # Create class-based subdirectories
            cnn_class_dir = os.path.join(output_dir_cnn, cnn_class)
            cnn_i_class_dir = os.path.join(output_dir_cnn_i, cnn_i_class)
            gmucnn_class_dir = os.path.join(output_dir_gmucnn, gmucnn_class)
            os.makedirs(cnn_class_dir, exist_ok=True)
            os.makedirs(cnn_i_class_dir, exist_ok=True)
            os.makedirs(gmucnn_class_dir, exist_ok=True)
    
            # Save the original image in the appropriate class folders if it doesn't already exist
            cnn_image_path = os.path.join(cnn_class_dir, image_name)
            cnn_i_image_path = os.path.join(cnn_i_class_dir, image_name)
            gmucnn_image_path = os.path.join(gmucnn_class_dir, image_name)
            
            if not os.path.exists(cnn_image_path):
                original_image.save(cnn_image_path)
            if not os.path.exists(cnn_i_image_path):
                original_image.save(cnn_i_image_path)
            if not os.path.exists(gmucnn_image_path):
                original_image.save(gmucnn_image_path)
    
    show_popup()

    
        
    
    
    
