# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 23:45:54 2026

@author: User
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def run_geometric_attention_experiment(
    V=1000,       # Vocabulary size
    d=32,         # Base dimension of tokens
    k=4,          # Order of the distance (k-space)
    d_primes=[16, 32, 68, 128, 256, 512],  # Test embedding sizes for standard Q/K
    steps=2500,   # Optimization steps
    lr=0.05
):
    print(f"--- Generating Geometric Token Space ---")
    print(f"Vocabulary Size (V): {V}")
    print(f"Base dim (d): {d}, Subspace order (k): {k}")
    print(f"Intrinsic parameters per token: {d + k}\n")

    torch.manual_seed(42)

    # 1. Generate intrinsic token representations
    
    # Draw v from a Uniform distribution [-1, 1] instead of Normal
    # This prevents v^k from exploding geometrically
    v = torch.rand(V, d) * 2.0 - 1.0  
    
    # W: The d x k key representation matrix. Columns are v^1, v^2, ..., v^k
    W = torch.stack([v ** p for p in range(1, k + 1)], dim=2)  # Shape: (V, d, k)
    
    # theta: The k-dimensional coefficients
    theta = torch.randn(V, k)
    
    # Compute X as a linear combination of W's columns
    X_raw = torch.bmm(W, theta.unsqueeze(2)).squeeze(2)  # Shape: (V, d)
    
    # Normalize X into the [-1, 1] hypercube (L_inf norm scaling)
    # Because we only apply a scalar division to each X_i, it remains exactly 
    # inside the k-dimensional subspace defined by W_i.
    max_vals = torch.max(torch.abs(X_raw), dim=1, keepdim=True).values
    X = X_raw / max_vals  # Now every coordinate of X is strictly between -1 and 1

    # 2. Compute the True Distance Matrix A
    # A_ij = || X_i - P_j X_i ||^2, where P_j is the projection matrix of W_j
    print("Computing target distance matrix A...")
    
    # Compute the pseudo-inverse and projection matrices for all keys
    W_pinv = torch.linalg.pinv(W)             # Shape: (V, k, d)
    P = torch.bmm(W, W_pinv)                  # Shape: (V, d, d)

    # Efficient vectorized computation of all pairwise distances
    # jmn = key j, row m, col n | in = query i, col n -> ijm (query i projected onto key j)
    X_proj = torch.einsum('jmn, in -> ijm', P, X)  # Shape: (V, V, d)
    
    # || X_i - P_j X_i ||^2
    diff = X.unsqueeze(1) - X_proj                 # Shape: (V, V, d)
    A = (diff ** 2).sum(dim=2)                     # Shape: (V, V)

    # Convert distance to similarity (so higher dot-product matches closer geometric proximity)
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
    run_geometric_attention_experiment()