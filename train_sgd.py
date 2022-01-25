import json
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from logistic_regression import LogisticRegression
from dataset_loader import BinaryDataset
from utils import prepare_data, change_format


logging.basicConfig(level=logging.INFO, format='%(message)s')


def train(model, x, y, criterion):
    model.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    return loss, output


BATCH_SIZEs = [4]

DATA_PATH = "../covtype.libsvm.binary.scale"

A, y, m, n = prepare_data(DATA_PATH)
X, y = change_format(A, y, m)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dataset = {"train": {"features": X_train, "labels": y_train},
           "test": {"features": X_test, "labels": y_test}}

data_for_train = BinaryDataset(dataset)

criterion = nn.BCELoss()

maxiter = 3632

all_losses = []

for BATCH_SIZE in BATCH_SIZEs:
    torch.manual_seed(0)
    net = LogisticRegression(55, 1)
    optimizer = SGD(net.parameters(), lr=0.01)

    data_train = DataLoader(dataset=data_for_train, batch_size=BATCH_SIZE, shuffle=False)

    weights = [net.fc.weight.detach().clone()]
    for bidx, batch in enumerate(data_train):
        if bidx == maxiter:
            break

        x_train, y_train = batch['inp'], batch['oup']
        loss, predictions = train(net, x_train, y_train, criterion)
        weights.append(net.fc.weight.detach().clone())

        logging.info('Step {} Loss : {}'.format((bidx + 1), loss))

    losses = []
    net.eval()
    with torch.no_grad():
        for W in weights:
            net.fc.weight = torch.nn.Parameter(W)

            pred = net(X_test).flatten()
            cur_loss = criterion(pred, y_test).item()
            losses.append(cur_loss)
    with open(f'SGD_batch_{BATCH_SIZE}.pickle', 'wb') as handle:
        pickle.dump(losses, handle)
    all_losses.append(losses)


plt.xlabel('Iteration')
plt.ylabel('Loss')
for i, losses in enumerate(all_losses):
    plt.plot(losses, label=f"batch size {BATCH_SIZEs[i]}")
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(500))
plt.grid(True, linestyle='-', color='0.75')
plt.legend()
plt.xlim(left=0)

plt.savefig('SGD_loss.png')

