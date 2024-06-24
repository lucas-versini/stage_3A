from models_NODE import *
from datasets_NODE import *

import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

dataset = 'intricated_crowns'
n_train = 500
d = 2
p = 32
lr = 1e-3
n_epochs = 50
order = 1

path_to_folder_plots = './plots/'

# Create folder if does not exist
if not os.path.exists(path_to_folder_plots):
    os.makedirs(path_to_folder_plots)
else:
    pass

input_dim, (X_train, y_train, X_test, y_test) = get_dataset(dataset, n_train)
X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)

model = get_model(p, input_dim, order).to(device)
n_params = get_n_params(model)

optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    model.train()
    _ = train(model, optimizer, criterion, X_train, y_train)

    model.eval()
    with torch.no_grad():
        _, train_loss, train_acc = evaluate(model, criterion, X_train, y_train)
        pred, test_loss, test_acc = evaluate(model, criterion, X_test, y_test)

        print(f"Epoch {epoch}/{n_epochs}. Train loss: {train_loss:.2f} | Train accuracy: {train_acc:.2f}", end = " ")
        print(f"| Test loss: {test_loss:.2f} | Test accuracy: {test_acc:.2f}")

print("\nTraining over.")

plot_result(X_train.cpu(), X_test.cpu(), y_test.cpu(), model, pred, path_to_folder_plots, "")