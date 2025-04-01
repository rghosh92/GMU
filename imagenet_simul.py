# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 19:13:25 2025

@author: User
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np
import time 

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
        X_cov_inv = torch.linalg.inv(X_cov)
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
            # A = (torch.exp(-err)-np.exp(-1.0))/(np.exp(0)-np.exp(-1.0))
            return torch.exp(-err)
            # return A-0.5
        else:
            if self.normalize:
                return 1-err
            else:
                return -err
            
# Define the ResNet-50 model
class ResNetModel(nn.Module):
    def __init__(self, num_classes=1000):  # Adjust the number of classes for ImageNet
        super(ResNetModel, self).__init__()
        self.resnet = resnet50(pretrained=True)  # Use pretrained weights
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Modify the classifier

    def forward(self, x):
        return self.resnet(x)


# Define ResNet-18 with SRNLayer as the first layer
class ResNetWithSRN(nn.Module):
    def __init__(self, num_classes=1000):  # Adjust number of classes if needed
        super(ResNetWithSRN, self).__init__()
        self.resnet = resnet50(pretrained=True)  # Load pretrained ResNet-18

        # Replace the first convolutional layer with SRNLayer
        self.resnet.conv1 = SRNLayer(
            input_channels=3,  # Input channels match RGB images
            output_channels=64,  # Same as the original ResNet conv1 output
            kernel_size=7,  # Original kernel size in ResNet conv1
            padding=3,  # Same padding as the original layer
            epsilon=0.0001,
            num_slices=1,  # Customize as needed
            degree=1,  # Customize as needed
            exponent=True,
            normalize=True
        )

        # Modify the final fully connected layer for the classification task
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


# Test ResNet-18 with its original architecture
model = ResNetModel(num_classes=1000)  # ImageNet has 1000 classes
dummy_input = torch.randn(10, 3, 224, 224)  # Batch size = 100, Channels = 3, Image size = 224x224
tt = time.time()
output = model(dummy_input)
print(f"Original ResNet-18 Inference Time: {time.time() - tt}")
print(f"Output shape: {output.shape}")  # Should be [100, 1000]

# Test ResNet-18 with SRNLayer
model = ResNetWithSRN(num_classes=1000)  # Replace first layer with SRNLayer
output = model(dummy_input)
tt = time.time()
output = model(dummy_input)
print(f"ResNet-18 with SRNLayer Inference Time: {time.time() - tt}")
print(f"Output shape: {output.shape}")  # Should be [100, 1000]
  # Should be [1, 1000]

