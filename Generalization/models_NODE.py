import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

#################
# Order 1 ODE
#################

class ODEFunc_order_1(nn.Module):
    def __init__(self, p, input_dim):
        super(ODEFunc_order_1, self).__init__()
        self.p = p
        self.input_dim = input_dim

        self.w_net = Network(1, 64, self.p * self.input_dim)
        self.a_net = Network(1, 64, self.p * self.input_dim)
        self.b_net = Network(1, 64, self.p)

    def forward(self, t, x):
        t = t.unsqueeze(0)
        w = self.w_net(t).reshape(self.input_dim, self.p)
        a = self.a_net(t).reshape(-1, self.p, self.input_dim)
        b = self.b_net(t).reshape(-1)
        inner_product = torch.matmul(a, x.unsqueeze(2)).squeeze(2)
        f_value = torch.matmul(torch.relu(inner_product + b), w.T)
        return f_value
    
class FullModel_order_1(nn.Module):
    def __init__(self, input_dim = 2, p = 16):
        super(FullModel_order_1, self).__init__()
        self.ode_func = ODEFunc_order_1(p, input_dim)
        self.proj = nn.Linear(2, 1)

    def forward(self, x):
        pred = odeint(self.ode_func, x, torch.tensor([0.0, 1.0]).to(device), method='dopri5', atol=1e-4, rtol=1e-4)[-1]
        return self.proj(pred)

#################
# Order 2 ODE
#################

# Define the ODE function, that returns the derivative of x(t)
class ODEFunc_order_2(nn.Module):
    def __init__(self, p, input_dim):
        super(ODEFunc_order_2, self).__init__()
        self.p = p
        self.input_dim = input_dim

        self.w_net = Network(1, 64, self.p * self.input_dim)
        self.a_net = Network(1, 64, self.p * self.input_dim)
        self.b_net = Network(1, 64, self.p)

    def forward(self, t, x):
        y, z = x[:, :self.input_dim], x[:, self.input_dim:]
        t = t.unsqueeze(0)
        w = self.w_net(t).reshape(self.input_dim, self.p)
        a = self.a_net(t).reshape(-1, self.p, self.input_dim)
        b = self.b_net(t).reshape(-1)
        inner_product = torch.matmul(a, y.unsqueeze(2)).squeeze(2)
        f_value = torch.matmul(torch.relu(inner_product + b), w.T)
        return torch.cat((z, -z + f_value), dim=1)

class FullModel_order_2(nn.Module):
    def __init__(self, input_dim = 2, p = 16):
        super(FullModel_order_2, self).__init__()
        self.input_dim = input_dim
        self.p = p
        self.ode_func = ODEFunc_order_2(p, input_dim)
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = torch.cat((x, torch.zeros_like(x)), dim=1)
        pred = odeint(self.ode_func, x, torch.tensor([0.0, 1.0]).to(device), method='dopri5', atol=1e-4, rtol=1e-4)[-1]
        pred = pred[:, :self.input_dim]
        return self.proj(pred)

#################
# Get model
#################

def get_model(p, input_dim, order = 1):
    if order == 1:
        return FullModel_order_1(input_dim, p)
    else:
        return FullModel_order_2(input_dim, p)

#################
# Training and Evaluation
#################

def train(model, optimizer, criterion, X_train, y_train):
    optimizer.zero_grad()
    pred = nn.Sigmoid()(model(X_train))
    loss = criterion(pred.squeeze(), y_train.squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, criterion, X_test, y_test):
    with torch.no_grad():
        pred = nn.Sigmoid()(model(X_test))
        loss = criterion(pred.squeeze(), y_test.squeeze()).item()
        pred_labels = (pred >= 0.5).float().squeeze()
        accuracy = (pred_labels == y_test).float().mean().item()
    return pred, loss, accuracy

#################
# Decision boundary
#################

def plot_result(X_train, X_test, y_test, model, pred, path, name):
    model.eval()

    # Generate a grid of points
    x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - 1, max(X_train[:, 0].max(), X_test[:, 0].max()) + 1
    y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - 1, max(X_train[:, 1].max(), X_test[:, 1].max()) + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

    # Predictions for the entire grid
    with torch.no_grad():
        grid_pred = nn.Sigmoid()(model(grid_tensor)).cpu().numpy().reshape(xx.shape)

    # Plotting
    plt.figure(figsize = (6, 5))
    plt.contourf(xx, yy, grid_pred, alpha = 0.8, cmap = plt.cm.coolwarm)
    plt.colorbar()
    plt.title('Decision boundary')
    plt.savefig(path + name + '_decision_boundary' + '.png')

    plt.figure(figsize = (6, 5))
    pred_labels = (pred >= 0.5).float().squeeze()
    misclassified_indices = np.where(pred_labels.cpu().detach().numpy().squeeze() != y_test.numpy())[0]
    plt.scatter(X_test[misclassified_indices, 0], X_test[misclassified_indices, 1], c = 'none', edgecolors = 'black', linewidths = 5)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=pred_labels.cpu().detach().numpy().squeeze(), cmap='coolwarm')
    plt.title('Predictions on the test set')
    plt.savefig(path + name + '_predictions' + '.png')

    plt.close('all')
