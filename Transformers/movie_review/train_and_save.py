from tqdm import tqdm

import pickle
import os

import matplotlib.pyplot as plt

from dataset import *
from model import *

import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

path_save = './results/'

if not os.path.exists(path_save):
    os.makedirs(path_save)

# Hyperparameters
lr = 1e-3
batch_size = 128
n_epochs = 10
num_layers = 6
d = 2

# Load the data
data_loader_train, data_loader_test, len_all_words, max_len, data_train, data_test, dic = load_data(batch_size = batch_size, path = "./IMDB.csv")
print("Data loaded")

# Create the model
model = CustomTransformer(num_words = len_all_words, d = d, num_layers = num_layers, beta = 1., share = False, value = "identity")
model.to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

train_losses = []
test_losses = []
train_accu = []
test_accu = []

for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    train_accuracy = 0
    total = 0
    for data, target in tqdm(data_loader_train, total = len(data_loader_train), desc = f'Epoch {epoch + 1}/{n_epochs}'):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += loss.item()
            train_accuracy += ((output > 0.5) == target).sum().item()
            total += len(target)
    train_losses.append(train_loss / len(data_loader_train))
    train_accu.append(train_accuracy / total)

    model.eval()
    test_loss = 0
    test_accuracy = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(data_loader_test, total = len(data_loader_test), desc = 'Test '):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target.float())
            test_loss += loss.item()
            test_accuracy += ((output > 0.5) == target).sum().item()
            total += len(target)
        test_losses.append(test_loss / len(data_loader_test))
        test_accu.append(test_accuracy / total)

    print(f'Epoch {epoch+1}, train_loss: {train_losses[-1]}, train_accu: {train_accu[-1]}, test_loss: {test_losses[-1]}, test_accu: {test_accu[-1]}')

# Save model and data

torch.save(model.state_dict(), os.path.join(path_save, 'model.pth'))
print('Model saved')

with open(os.path.join(path_save, 'results_training.pkl'), 'wb') as f:
    pickle.dump((train_losses, test_losses, train_accu, test_accu), f)
with open(os.path.join(path_save, 'embeddings.pkl'), 'wb') as f:
    pickle.dump((data_loader_train, data_loader_test, len_all_words, max_len, data_train, data_test, dic), f)