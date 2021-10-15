import json
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from logistic_regression import LogisticRegression
from dataset_loader import BinaryDataset
from utils import prepare_data, change_format
from vaidya import vaidya_for_logreg, get_init_polytope_square


logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_loss(model, x, y, criterion):
    model.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    return loss


def oracle(model, x, y, criterion, x_k):
    model.fc.weight = nn.Parameter(torch.from_numpy(x_k.astype(np.float32)), requires_grad=True)
    loss = get_loss(model, x, y, criterion)
    return model.fc.weight.grad.detach().numpy(), loss


EPOCHS = 3
BATCH_SIZE = 25000
DATA_PATH = "../Downloads/covtype.libsvm.binary.scale"

A, y, m, n = prepare_data(DATA_PATH)
X, y = change_format(A, y, m)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
dataset = {"train": {"features": X_train, "labels": y_train},
           "test": {"features": X_test, "labels": y_test}}

data_for_train = BinaryDataset(dataset)
data_for_test = BinaryDataset(dataset, mode="test")
data_train = DataLoader(dataset=data_for_train, batch_size=BATCH_SIZE, shuffle=False)
data_test = DataLoader(dataset=data_for_test, batch_size=BATCH_SIZE, shuffle=False)
criterion = nn.BCELoss()
loss_values = []

n = 55
np.random.seed(0)
x_0 = np.random.randn(n, 1)
x_0 /= np.linalg.norm(x_0)

R = 10.
A_0, b_0 = get_init_polytope_square(n, R)

K = 100

eps = 0.1
eta = 100

net = LogisticRegression(55, 1)
loss_values = []

for epoch in range(EPOCHS):

    W_0 = net.fc.weight.detach().numpy()
    W, epoch_losses = vaidya_for_logreg(A_0, b_0, W_0.T, eps, eta, oracle, get_loss,
                                        net, data_train, criterion, stepsize=0.18, verbose=False)
    net.fc.weight = torch.nn.Parameter(torch.from_numpy(W.T.astype(np.float32)), requires_grad=True)
    loss_values += epoch_losses

plt.title('BCE Loss for Vaidya method')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.plot(range(1, len(loss_values) + 1), loss_values, 'r')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
plt.grid(True, linestyle='-', color='0.75')
plt.savefig('vaidya_loss.png')

with open("vaidya_loss.json", 'w') as f:
    json.dump(loss_values, f, indent=2)
