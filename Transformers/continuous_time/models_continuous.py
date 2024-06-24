import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

class ODEFunc(nn.Module):
    def __init__(self, d, hidden_dim, beta = 1.0, random_init = False, n = None):
        super(ODEFunc, self).__init__()
        self.d = d
        self.hidden_dim = hidden_dim
        self.beta = beta

        if random_init:
            if n is None:
                raise ValueError("n must be provided")
            self.K = nn.Parameter(2 / np.sqrt(n) * torch.randn(self.hidden_dim, self.d)).to(device)
            self.Q = nn.Parameter(2 / np.sqrt(n) * torch.randn(self.hidden_dim, self.d)).to(device)
            self.V = self.Q.T @ self.K
        else:
            self.K = nn.Parameter(torch.eye(self.hidden_dim)).to(device)
            self.Q = nn.Parameter(torch.eye(self.hidden_dim)).to(device)
            self.V = nn.Parameter(torch.eye(self.d)).to(device)
    
    def forward(self, t, x):
        # Shape of x: batch, n, d
        value_ = torch.einsum('ij,bnj->bni', self.V, x) # n, d
        key_ = torch.einsum('ij,bnj->bni', self.K, x) # n, hidden_dim
        query_ = torch.einsum('ij,bnj->bni', self.Q, x) # n, hidden_dim
        attention = self.beta * torch.einsum('bni,bmi->bnm', key_, query_) # b, n, n
        attention = F.softmax(attention, dim = 1) # b, n, n

        temp = torch.einsum('bnm,bnj->bmj', attention, value_) # n, d
        temp = temp - torch.einsum('bnj,bnj->bn', temp, x)[:, :, None] * x # n, d

        return temp

class FullModel(nn.Module):
    def __init__(self, d, hidden_dim, beta = 1.0, random_init = False, n = None):
        super(FullModel, self).__init__()
        self.d = d
        self.hidden_dim = hidden_dim
        self.beta = beta

        self.odefunc = ODEFunc(d, hidden_dim, beta, random_init, n)
    
    def forward(self, x, t = None):
        return odeint(self.odefunc, x, t, method = 'dopri5')

def dynamic_gamma_beta(t, y, beta, n):
    return 2 * np.exp(beta * y) * (1 - y) * ((n - 1) * y + 1) / (np.exp(beta) + (n - 1) * np.exp(beta * y))
