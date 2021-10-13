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
    get_loss(model, x, y, criterion)
    return model.fc.weight.grad.detach().numpy()


def train(model, x, y, criterion):
    W_0 = model.fc.weight.detach().numpy()
    new_weights = vaidya_for_logreg(A_0, b_0, W_0.T, eps, eta, K, oracle, model, x, y, criterion, stepsize=0.18, verbose=False)[-1]
    model.fc.weight = torch.nn.Parameter(torch.from_numpy(new_weights.T.astype(np.float32)), requires_grad=True)
    loss = get_loss(model, x, y, criterion)

    #print(loss)

    new_output = model(x)
    return loss, new_output


EPOCHS = 1
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

for epoch in range(EPOCHS):

    epoch_loss = 0

    for bidx, batch in enumerate(data_train):

        x_train, y_train = batch['inp'], batch['oup']
        loss, predictions = train(net, x_train, y_train, criterion)
        loss_values.append(float(loss))
        epoch_loss += loss

        logging.info('Step {} Loss : {}'.format((bidx + 1), loss))

    logging.info('Epoch {} Loss : {}'.format((epoch + 1), epoch_loss))

plt.title('BCE Loss for Vaidya method')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.plot(range(1, len(loss_values) + 1), loss_values, 'r')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.grid(True, linestyle='-', color='0.75')
plt.savefig('vaidya_loss.png')

with open("vaidya_loss.json", 'w') as f:
    json.dump(loss_values, f, indent=2)
