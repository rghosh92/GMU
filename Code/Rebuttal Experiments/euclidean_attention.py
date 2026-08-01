# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 00:44:34 2026

@author: User
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def run_euclidean_attention_experiment(
    V=1000,       # Vocabulary size
    d=32,         # Dimension of tokens
    d_primes=[16, 32,33, 68],  # Test embedding sizes for standard Q/K
    steps=2500,   # Optimization steps
    lr=0.05
):
    print(f"--- Generating Euclidean Token Space ---")
    print(f"Vocabulary Size (V): {V}")
    print(f"Token dimension (d): {d}")
    print(f"Intrinsic parameters per token: {d}\n")

    torch.manual_seed(42)

    # 1. Generate intrinsic token representations
    # Each token is just a single d-dimensional point
    # Bounded in [-1, 1] hypercube
    X = torch.rand(V, d) * 2.0 - 1.0

    # 2. Compute the True Distance Matrix A
    # A_ij = || X_i - X_j ||^2
    print("Computing target symmetric Euclidean distance matrix A...")
    
    # diff shape: (V, V, d) where diff[i, j] = X[i] - X[j]
    diff = X.unsqueeze(1) - X.unsqueeze(0)
    A = (diff ** 2).sum(dim=2)  # Shape: (V, V)

    # Convert distance to similarity (so higher dot-product matches closer proximity)
    Target_Sim = -A
    
    # Row-center the target to ignore query-specific offsets (softmax shift invariance)
    Target_centered = Target_Sim - Target_Sim.mean(dim=1, keepdim=True)

    # 3. Experiment: Learn Q and K embeddings to match Target_centered
    print("\n--- Training Standard QK^T Factorizations ---")
    
    results = {}
    
    for d_prime in d_primes:
        # Initialize standard Query and Key embeddings
        Q = nn.Parameter(torch.randn(V, d_prime) * 0.01)
        K = nn.Parameter(torch.randn(V, d_prime) * 0.01)
        
        optimizer = optim.Adam([Q, K], lr=lr)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Compute standard dot-product attention logits
            M = Q @ K.T
            
            # Row-center to enforce shift-invariant difference matching
            M_centered = M - M.mean(dim=1, keepdim=True)
            
            # MSE loss between the centered matrices
            loss = F.mse_loss(M_centered, Target_centered)
            
            loss.backward()
            optimizer.step()
            
        results[d_prime] = loss.item()
        
        # Check relative error to see how well it fits compared to the scale of the target
        target_var = torch.var(Target_centered).item()
        explained_variance = 1.0 - (loss.item() / target_var)
        
        print(f"d' = {d_prime:<4} | Final MSE: {loss.item():<8.4f} | Explained Variance: {explained_variance * 100:>6.2f}%")

    return results

if __name__ == "__main__":
    run_euclidean_attention_experiment()