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
from torch.utils.data import TensorDataset
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

from torch.utils.data import Dataset, DataLoader

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


import torch
from torch.utils.data import Dataset, DataLoader

class PrecomputedDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        x = self.inputs[index]
        if self.transform is not None:
            x = self.transform(x)
        y = self.labels[index]
        return x, y





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


    
class SimpleGMUfc():
    def __init__(self,input_channels, output_channels, threshold, num_slices=2):
        super(SimpleGMUfc, self).__init__()
        
        self.weights = torch.zeros(output_channels, input_channels,num_slices)
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        self.threshold = threshold
        
        for s in range(num_slices):
            # assign a unique input channel index for this slice
            self.weights[:, s, s] = 1.0
        

    def output(self, y):
            # print(self.weights.shape)
            y = y.squeeze()
            if len(y.shape)==1:
                y = y.unsqueeze(0)
                      
            y = y.unsqueeze(1).repeat(1,self.weights.shape[0],1)
            
            X = self.weights

            
            X_cov = torch.einsum('bij,bki->bjk', X, X.permute(0,2,1))
            X_cov_inv = torch.linalg.inv(X_cov)
            M = torch.einsum('bij,bkj->bik', X_cov_inv, X)
            
            
            W = torch.einsum('ijk,bik->ijb',M,y)
            
            pred_final = torch.einsum('bij,bjk->bik', X, W)
            
            pred_final = pred_final.permute(2,0,1)
            
            dist = torch.sqrt(torch.mean((y-pred_final)**2,dim=2))
            # err = err.unsqueeze(2).unsqueeze(3)
            return dist>self.threshold
            
        
        
class uRBF(nn.Module):
    def __init__(self,input_channels, output_channels, num_slices):
        super(uRBF, self).__init__()
        
        self.weight_bias = torch.nn.Parameter(torch.zeros(output_channels, input_channels))
        
        self.sigma = torch.nn.Parameter(torch.ones(1,output_channels,1,1))
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_slices = num_slices
        
        self.init_weights()
        
    def init_weights(self):
        self.weight_bias.data.uniform_(-0.01, 0.01)
        self.sigma.data.uniform_(0.5, 1.5)


    def forward(self, y):
        y = y.squeeze()
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
    
        # expand input to match weight_bias
        y = y.unsqueeze(1).repeat(1, self.weight_bias.shape[0], 1)
        y = y - self.weight_bias.unsqueeze(0).repeat(y.shape[0], 1, 1)
    
        # squared distance
        err = torch.sum(y**2, dim=2)
    
        # divide by 2 * sigma^2
        out = torch.exp(-err / (2 * self.sigma.squeeze()**2))
    
        return out

        
        
class RBFMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                  num_slices, normalize=True, use_dropout=True):
        super(RBFMLP, self).__init__()
        
        # Hidden GMU layer
        self.rbf_in = uRBF(input_dim, hidden_dim,num_slices)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(1)
        # Dropout
        # self.drop = nn.Dropout(0.2) if use_dropout else nn.Identity()
        
        # Final fully connected classifier
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        # Hidden GMU + BN
        out = self.rbf_in(x)
        out = self.bn1(out)
        
        # Dropout
        # out = self.drop(out)
        
        # Final linear output (logits)
        out = self.fc_out(out)
        
        return out




def train_network_normal(net, trainloader, init_rate, epochs, weight_decay, device="cuda"):
    net = net.to(device)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=init_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")  # sum so we can weight manually

    all_train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        total_samples = 0

        for inputs, labels in trainloader:
            labels =labels.float().view(-1, 1) 

            optimizer.zero_grad()
            logits = net(inputs)

            # sum reduction gives total loss over batch
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # accumulate weighted loss
            batch_size = labels.size(0)
            epoch_loss += loss.item()
            total_samples += batch_size

        # weighted average loss over all samples
        avg_loss = epoch_loss / total_samples
        all_train_losses.append(avg_loss)

    print("Final weighted train loss:", all_train_losses[-1])
    return net, all_train_losses



def test_network(net, testloader):
    net.eval()
    TP = TN = FP = FN = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            labels = labels.float().view(-1, 1)

            logits = net(inputs)                  # (batch, 1)
            probs = torch.sigmoid(logits)         # convert to probabilities
            preds = (probs > 0.5).long()          # threshold at 0.5

            # confusion matrix components
            TP += ((preds == 1) & (labels == 1)).sum().item()
            TN += ((preds == 0) & (labels == 0)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()

    # avoid division by zero
    recall_pos = TP / (TP + FN) if (TP + FN) > 0 else 0
    recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0

    balanced_accuracy = 0.5 * (recall_pos + recall_neg)
    return balanced_accuracy




import seaborn as sns


import math


def v_unit_k(k):
    return (math.pi ** (k / 2)) / math.gamma(k / 2 + 1)

def balanced_error_estimate(d, k, R, n):

    v_k_1 = v_unit_k(k)
    denominator = (R**2) * np.power(n * v_k_1, 2/k)
    x = 1 / denominator
    
    if x >= 1:
        print(x)
        return 0.5
    
    vol_ratio = np.power(1 - x, (d - k) / 2)
    
    epsilon = 0.5 * (1 - vol_ratio) 
    
    return epsilon


def v_unit_k(k):
    """Standard volume of a unit k-ball."""
    return (math.pi ** (k / 2)) / math.gamma(k / 2 + 1)

def balanced_error_mlp_estimate(d, k, N):
    """
    Estimates balanced error for an MLP approximating a k-order GMU.
    
    Formula: 
    epsilon = [m(m-1) / 4(m+1)] * [m*Vm / Vm-1]^(2/(m-1)) * N^(-2/(m-1))
    where m = d - k.
    """
    m = d - k
    
    # If m=1, the boundary is just two points (a 1D sphere), 
    # which an MLP fits perfectly with 2 neurons.
    if m <= 1:
        return 0.0
    
    # 1. Geometry Factor: m(m-1) / 4(m+1)
    geo_factor = (m * (m - 1)) / (4 * (m + 1))
    
    # 2. Volume Ratio: (m * V_m) / V_{m-1}
    # We use log-gamma to calculate this safely for high dimensions
    # log(Vm) = (m/2)log(pi) - lgamma(m/2 + 1)
    def log_v_unit(n):
        return (n / 2) * math.log(math.pi) - math.lgamma(n / 2 + 1)
    
    log_ratio = math.log(m) + log_v_unit(m) - log_v_unit(m - 1)
    ratio_val = math.exp(log_ratio)
    
    # 3. Combined Exponent: 2 / (m - 1)
    exponent = 2 / (m - 1)
    
    # 4. Capacity Term: (ratio_val / N)^exponent
    # This combines the volume ratio and N into the power term
    capacity_term = math.pow(ratio_val / N, exponent)
    
    # 5. Final result
    epsilon = geo_factor * capacity_term
    
    # Cap at 0.5 (random guessing limit)
    return min(0.5, epsilon)


import math

def balanced_error_mlp_tangent_corrected(d, k, N):
    """
    Balanced error estimate for an MLP approximating a k-order GMU.
    Uses the corrected tangent-bound derivation:

    epsilon = [m(m-1) / 4(m+1)] * [ (m * V_m) / V_{m-1} ]^(2/(m-1)) * N^(-2/(m-1))

    where m = d - k.
    """
    m = d - k
    
    # If codimension is 1 or less, approximation is effectively perfect
    if m <= 1:
        return 0.0
    
    # Volume of unit n-ball: V_n = pi^(n/2) / Gamma(n/2 + 1)
    def unit_vol(n):
        return math.pi**(n/2) / math.gamma(n/2 + 1)
    
    Vm = unit_vol(m)
    Vm_minus_1 = unit_vol(m - 1)
    
    # Geometric coefficient
    lead_coeff = (m * (m - 1)) / (4 * (m + 1))
    
    # Volume ratio term
    vol_ratio = (m * Vm) / Vm_minus_1
    
    # Exponent
    exponent = 2 / (m - 1)
    
    # Capacity term
    capacity_term = (vol_ratio ** exponent) * (N ** (-exponent))
    
    # Final epsilon
    epsilon = lead_coeff * capacity_term
    
    # Saturation at 0.5 (random guess limit)
    return min(0.5, epsilon)



from scipy.special import gammaln

def compute_gmu_error(m, N):
    """
    Computes the balanced error epsilon_updated = epsilon / (1 + 2*epsilon)
    for a k-order GMU where m = d - k.
    
    Args:
        m (int): Effective dimensionality (d - k). Must be > 1.
        N (int): Number of hidden units (neurons).
        
    Returns:
        float: The updated balanced error bound.
    """
    if m <= 1:
        return 0.5 # The geometric approximation assumes m > 1
    
    # 1. Compute Log(V_n) for n = m and n = m-1
    # Formula: V_n = pi^(n/2) / Gamma(n/2 + 1)
    def log_unit_volume(n):
        return (n / 2) * np.log(np.pi) - gammaln(n / 2 + 1)
    
    log_Vm = log_unit_volume(m)
    log_Vm1 = log_unit_volume(m - 1)
    
    # 2. Compute the Log of the Volume Ratio term: (m * Vm / Vm-1)^(2 / (m-1))
    # log_ratio = [ log(m) + log_Vm - log_Vm1 ] * (2 / (m-1))
    log_vol_term = (np.log(m) + log_Vm - log_Vm1) * (2 / (m - 1))
    
    # 3. Compute the Log of the Capacity term: N^(-2 / (m-1))
    log_capacity_term = (-2 / (m - 1)) * np.log(N)
    
    # 4. Compute the Geometric Coefficient
    # coeff = m(m-1) / 4(m+1)
    coeff = (m * (m - 1)) / (4 * (m + 1))
    
    # 5. Combine to find base epsilon
    # epsilon = coeff * exp(log_vol_term + log_capacity_term)
    epsilon = coeff * np.exp(log_vol_term + log_capacity_term)
    
    # 6. Apply your updated normalization: epsilon / (1 + 2*epsilon)
    epsilon_updated = epsilon / (1 + 2 * epsilon)
    
    return epsilon_updated




# def balanced_error_mlp_tangent(d, k, N):
#     """
#     Optimistic balanced error estimate for an MLP approximating a k-order GMU.
#     Uses the tangent (circumscribed) derivation:
    
#     epsilon = [(m-1) / 4(m+1)] * [(m * Vm) / Vm_minus_1]^(2/(m-1)) * N^(-2/(m-1))
    
#     where m = d - k.
#     """
#     m = d - k
    
#     # If the codimension is 1 or less, the approximation is effectively perfect
#     if m <= 1:
#         return 0.0
    
#     # 1. Lead Coefficient: (m - 1) / (4 * (m + 1))
#     # This term approaches 0.25 as m increases.
#     lead_coeff = (m - 1) / (4 * (m + 1))
    
#     # 2. Volume Log-Ratio Calculation
#     # We compute log(Vm) - log(Vm-1) to handle large dimensions safely.
#     # log(Vn) = (n/2)*log(pi) - log_gamma(n/2 + 1)
#     def log_unit_vol(n):
#         return (n / 2) * math.log(math.pi) - math.lgamma(n / 2 + 1)
    
#     # Log of [ (m * Vm) / Vm_minus_1 ]
#     log_ratio = math.log(m) + log_unit_vol(m) - log_unit_vol(m - 1)
    
#     # 3. Exponent: 2 / (m - 1)
#     # This is the "Curse of Dimensionality" term.
#     exponent = 2 / (m - 1)
    
#     # 4. Capacity Term: ( (m * Vm / Vm_minus_1) / N )^exponent
#     # Using log space for power: exp(log_term * exponent)
#     log_capacity_term = (log_ratio - math.log(N)) * exponent
#     capacity_term = math.exp(log_capacity_term)
    
#     # 5. Final Balanced Error
#     epsilon = lead_coeff * capacity_term
    
#     epsilon = epsilon/(1+(2*epsilon))
#     # Saturation at 0.5 (random guess limit)
#     return min(0.5, epsilon)

# def balanced_error_mlp_normalized(d, k, N):
#     """
#     Computes the normalized balanced error for an MLP on a k-order GMU.
    
#     Formula:
#     R = [m(m-1) / 2(m+1)] * [ (m*Vm / Vm_minus_1) / N ]^(2/(m-1))
#     epsilon = R / (2 * (1 + R))
    
#     Args:
#         d (int): Total input dimensions.
#         k (int): Active subspace dimensions (GMU order).
#         N (int): Number of hidden units (neurons).
#     """
#     m = d - k
    
#     # If the null space is 1D or less, the polytope fits perfectly.
#     if m <= 1:
#         return 0.0
    
#     # 1. Lead Coefficient: m(m-1) / 2(m+1)
#     # This represents the geometric scaling of the discrepancy.
#     lead_coeff = (m * (m - 1)) / (2 * (m + 1))
    
#     # 2. Volume Calculations (Numerical Stability via Log-Gamma)
#     def log_unit_vol(n):
#         # Volume of n-dimensional unit ball
#         return (n / 2) * math.log(math.pi) - math.lgamma(n / 2 + 1)
    
#     # Log of the ratio: m * V_m / V_{m-1}
#     log_v_ratio = math.log(m) + log_unit_vol(m) - log_unit_vol(m - 1)
    
#     # 3. The "Curse of Dimensionality" Exponent
#     exponent = 2 / (m - 1)
    
#     # 4. Calculate the Volume Ratio R
#     # R = lead_coeff * ( (m * V_m / V_{m-1}) / N )^exponent
#     log_R = math.log(lead_coeff) + (log_v_ratio - math.log(N)) * exponent
#     R = math.exp(log_R)
    
#     # 5. Final Normalized Error
#     # This ensures epsilon stays in [0, 0.5]
#     epsilon = R / (2 * (1 + R))
    
#     return epsilon


# --- Example for d=50, k=3, N=500 ---
# error = balanced_error_mlp_tangent(50, 3, 500)
# print(f"Predicted Balanced Error: {error:.4%}")

# --- Validation for your specific setup ---
# d = 50, k = 3, N = 500
# m = 47
# epsilon = balanced_error_mlp_estimate(50, 3, 500)
# print(f"Predicted Error: {epsilon:.4f}")

# Example usage:
# print(balanced_error_mlp(d=50, k=3, R=0.25, N=500))

# def balanced_error_estimate(d, k, R, n):
#     # --- 1. Constants and Volumes ---
#     v_k_1 = v_unit_k(k)
#     v_d_1 = v_unit_k(d)
#     v_dk_1 = v_unit_k(d - k)
    
#     # Critical n: when n spheres of radius R fill the k-dimensional unit lattice
#     # V_k(R) = v_k_1 * R^k
#     n_critical = 1 / (v_k_1 * (R**k))
#     # --- 2. The Logic Switch ---
#     if n < n_critical:
#         print('he')
#         # SPARSE REGIME: No significant overlaps. 
#         # Error is based on the sum of individual volumes.
#         ratio_single = (v_d_1 / v_dk_1) * (R**k)
#         epsilon = 0.5 * (1 - n * ratio_single)
#     else:
#         print('ha')
#         # SATURATED REGIME: Spheres overlap to form a slab.
#         # We use the threshold height h* derived from the Poisson process.
#         x = 1 / ((R**2) * np.power(n * v_k_1, 2/k))
        
#         if x < 1:
#             vol_ratio_saturated = np.power(1 - x, (d - k) / 2)
#             epsilon = 0.5 * (1 - vol_ratio_saturated)
#         else:
#             # If the density is still physically too low to bridge the height
#             epsilon = 0.5

#     return max(0.0, min(0.5, epsilon))
    
    # a = input("")
        


def sample_hypersphere(n_samples, dim, r1=0.0, r2=1.0, device='cuda'):
    x = torch.randn(n_samples, dim, device=device)
    u = x / x.norm(dim=1, keepdim=True)
    U = torch.rand(n_samples, device=device)
    r = (r1**dim + U * (r2**dim - r1**dim))**(1.0/dim)
    return u * r.unsqueeze(1)


def generate_dataset(n_samples, n_dim, num_slices, threshold1, threshold2, device='cuda'):
    n0 = n_samples // 2
    n1 = n_samples - n0

    slices0 = torch.rand(n0, num_slices, device=device) - 0.5
    slices1 = torch.rand(n1, num_slices, device=device) - 0.5

    rest0 = sample_hypersphere(n0, n_dim - num_slices, r1=0.0, r2=threshold1, device=device)
    rest1 = sample_hypersphere(n1, n_dim - num_slices, r1=threshold1, r2=threshold2, device=device)

    inputs0 = torch.cat([slices0, rest0], dim=1)
    inputs1 = torch.cat([slices1, rest1], dim=1)

    inputs = torch.cat([inputs0, inputs1], dim=0)
    labels = torch.cat([torch.zeros(n0, device=device), torch.ones(n1, device=device)], dim=0)

    perm = torch.randperm(n_samples, device=device)
    return inputs[perm], labels[perm]

def gen_loaders(n_samples, n_dim, batch_size, num_slices, threshold1, device='cuda'):
    m = n_dim - num_slices

    threshold2 = threshold1 * (2**(1.0 / m))
    # print(threshold2)
    
    inputs_train, labels_train = generate_dataset(n_samples, n_dim, num_slices, threshold1, threshold2, device=device)
    inputs_test, labels_test = generate_dataset(n_samples, n_dim, num_slices, threshold1, threshold2, device=device)
    
    # print(labels_train.mean())
    # print(labels_test.mean())
    
    dataset_train = PrecomputedDataset(inputs_train, labels_train)
    dataset_test = PrecomputedDataset(inputs_test, labels_test)

    gen = torch.Generator(device=device)

    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                             generator=gen, num_workers=0)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                            generator=gen, num_workers=0)

    return trainloader, testloader



def balanced_error_secant(d, k, N):
    """
    Computes the balanced error epsilon for an MLP emulating a k-GMU
    using the refined Secant (Intersection) Bound.
    """
    m = d - k
    if m <= 1: 
        return 0.0
    
    # 1. Intersection Constant: [m(m-1) / (m+1)^2] * [(m-1)/(m+1)]^((m-1)/2)
    # This term converges to 1/e (~0.367) as m increases.
    term_a = (m * (m - 1)) / ((m + 1) ** 2)
    term_b_base = (m - 1) / (m + 1)
    log_term_b = ((m - 1) / 2) * math.log(term_b_base)
    
    log_intersection_constant = math.log(term_a) + log_term_b
    
    # 2. Volume term: (m * Vm / Vm-1)
    def log_unit_vol(n):
        # Log of the volume of an n-dimensional unit ball
        return (n / 2) * math.log(math.pi) - math.lgamma(n / 2 + 1)
    
    # log of (m * Vm / Vm-1)
    log_v_ratio = math.log(m) + log_unit_vol(m) - log_unit_vol(m - 1)
    
    # 3. Capacity Term: N^(-2/(m-1))
    exponent = 2 / (m - 1)
    
    # 4. Combine all components in log space
    # log(epsilon) = log(C_int) + exponent * log(V_ratio) - exponent * log(N)
    log_epsilon = log_intersection_constant + exponent * (log_v_ratio - math.log(N))
    
    epsilon_raw = math.exp(log_epsilon)
    
    # 5. Physical Saturation
    # In practice, balanced error for binary classification is capped at 0.5.
    # We use a smooth saturation (1 - exp) or a simple min(0.5).
    # To stay strictly true to your derivation, we use min.
    return min(epsilon_raw, 0.5)

# Example Usage:
# d=10, k=2 (m=8), N=1000
# print(balanced_error_secant(10, 2, 1000))


def balanced_error_final(d, k, N):
    m = d - k
    if m <= 1:
        return 0.0

    # A = m(m-1) / (4(m+1))
    A = (m * (m - 1)) / (4 * (m + 1))

    # log volume of unit n-ball
    def log_unit_vol(n):
        return (n / 2) * math.log(math.pi) - math.lgamma(n / 2 + 1)

    # B = (m Vm / V(m-1))^(2/(m-1))
    log_ratio = math.log(m) + log_unit_vol(m) - log_unit_vol(m - 1)
    exponent = 2 / (m - 1)
    B = math.exp(log_ratio * exponent)

    # C = N^(-2/(m-1))
    C = N ** (-2 / (m - 1))

    # epsilon = A * B * C
    epsilon = A * B * C
    return min(0.5,epsilon)


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
    
    
    n_samples = 10000
    n_dim = 10
    threshold1 = 0.25
    # threshold2 = 0.29
    # hidden_dim_range = np.arange(10,200,10)
    num_slices = [0,1,2,3,4,5,6]
    N_list = [1,10,50,100,200,500]
    # N_list = [1]
    output_dim = 1
    # Initialize your label generator
    
    


    #Training params
    init_rate = 0.0005
    total_epoch = 100
    batch_size = 400
    decay_normal = 0 
    
    
    final_errs = []
    final_ests = []
    # balan = [] 
    
    # Test one batch
    trainloader_list = []
    testloader_list = []
    
    for i in range(len(num_slices)):
        trainloader, testloader = gen_loaders(n_samples, n_dim, batch_size, num_slices[i], threshold1)
        trainloader_list.append(trainloader)
        testloader_list.append(testloader)
    
    
    for N in N_list:
        err_list = []
        est_list = [] 
        for i in range(len(num_slices)):
            
            # trainloader, testloader = gen_loaders(n_samples, n_dim,batch_size,num_slices[i], threshold)
            # trainloader, testloader = gen_loaders(n_samples, n_dim, batch_size, num_slices[i], threshold1, threshold2)
            
            net_rbf  = SimpleMLP(n_dim, N, output_dim)
            net_rbf,all_losses = train_network_normal(net_rbf,trainloader_list[i],
                                                        init_rate,total_epoch,decay_normal)
                    
            acc =  test_network(net_rbf,testloader_list[i])
            # if num_slices[i] > 0:
            est = balanced_error_secant(n_dim,num_slices[i],N)
            # else:
            # est = 0 
                
            print(est)
            # balan.append(est)
            print('Test Error MLP-' + str(N)+' :', 1-acc)
            err_list.append(1-acc)
            est_list.append(est)
            
            
        final_errs.append(err_list)
        final_ests.append(est_list)


    
    np.save("final_errs_mlp.npy", np.array(final_errs))
    np.save("final_ests_mlp.npy", np.array(final_ests))


        
    
    
    # plt.plot(balan)
        
            
    # corruptions = ['general']

    
     
    # net = SimpleMLP(input_channels, hidden_dim, output_dim)
    # net,all_losses = train_network_normal(net,trainloader,testloader,my_dataset_test.labels, init_rate,total_epoch,decay_normal)
    
    # acc =  test_network(net, testloader, my_dataset_test.labels, 0)
    # print('Test Accuracy MLP:', acc)
    
        
        # accor = test_network_corruptions(net, testloader, corruptions,atanh_convert)
        # average_corruption_accuracies.append(accor)
    
        
        # test_network_corruptions_with_bn_adaptation(net, testloader, corruptions)    


    
    
