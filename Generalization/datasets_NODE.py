import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#################
# Moon dataset
#################

def generate_classification_data_moon(n_train = 200, n_test = 1000):
    n_samples = n_train + n_test
    test_size = n_test / n_samples
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

#################
# Two circles dataset
#################

def generate_crown(n, r_min = 0, r_max = 1, theta_max = 2 * np.pi, a = 0, b = 0):
    radius = np.random.uniform(r_min, r_max, n)
    theta = np.random.uniform(0, theta_max, n)
    X = np.array([a + radius * np.cos(theta), b + radius * np.sin(theta)]).T
    y = np.zeros(n)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def generate_classification_intricated_crowns(n_train = 200, n_test = 1000, r_min_1 = 0, r_max_1 = 1, r_min_2 = 2, r_max_2 = 3, theta_max = 2 * np.pi, a = 0, b = 0):
    n = n_train + n_test

    X1, y1 = generate_crown(n_train // 2, r_min_1, r_max_1, theta_max, a=a, b=b)
    X2, y2 = generate_crown(n_train//2, r_min_2, r_max_2, theta_max, a=a, b=b)
    X_train = torch.cat([X1, X2], dim=0)
    y_train = torch.cat([y1, y2 + 1], dim=0)

    X1, y1 = generate_crown(n_test // 2, r_min_1, r_max_1, a=a, b=b)
    X2, y2 = generate_crown(n_test//2, r_min_2, r_max_2, a=a, b=b)
    X_test = torch.cat([X1, X2], dim=0)
    y_test = torch.cat([y1, y2 + 1], dim=0)

    return X_train, y_train, X_test, y_test

#################
# Circle dataset
#################

def generate_classification_data_circle(n_train = 200, n_test = 1000):
    # Disk of radius 1
    n_samples = n_train + n_test
    test_size = n_test / n_samples
    X = np.random.rand(n_samples, 2) * 3. - 1.5
    y = np.linalg.norm(X, axis=1) < 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

#################
# Four circles intricated
#################

def generate_classification_data_four_crowns(n_train = 200, n_test = 1000, a_1 = 0, b_1 = 0, a_2 = 6, b_2 = 2):
    X_train_1, y_train_1, X_test_1, y_test_1 = generate_classification_intricated_crowns(n_train, n_test, a=a_1, b=b_1)
    X_train_2, y_train_2, X_test_2, y_test_2 = generate_classification_intricated_crowns(n_train, n_test, a=a_2, b=b_2)
    X_train = torch.cat([X_train_1, X_train_2], dim=0)
    y_train = torch.cat([1 - y_train_1, y_train_2], dim=0)
    X_test = torch.cat([X_test_1, X_test_2], dim=0)
    y_test = torch.cat([1 - y_test_1, y_test_2], dim=0)
    return X_train, y_train, X_test, y_test

#################
# Get dataset
#################

def get_dataset(name, n_train):
    n_test = 1000
    if name == 'intricated_crowns':
        return 2, generate_classification_intricated_crowns(n_train, n_test)
    elif name == 'intricated_crowns_cut':
        return 2, generate_classification_intricated_crowns(n_train, n_test, theta_max = np.pi * 5 / 3)
    elif name == 'moon':
        return 2, generate_classification_data_moon(n_train, n_test)
    elif name == 'circle':
        return 2, generate_classification_data_circle(n_train, n_test)
    elif name == 'four_crowns':
        return 2, generate_classification_data_four_crowns(n_train, n_test)
    else:
        raise ValueError('Unknown dataset name')
