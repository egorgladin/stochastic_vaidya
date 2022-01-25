import json
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import time

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


BATCH_SIZEs = [128, 256]

DATA_PATH = "../covtype.libsvm.binary.scale"

A, y, m, n = prepare_data(DATA_PATH)
X, y = change_format(A, y, m)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dataset = {"train": {"features": X_train, "labels": y_train},
           "test": {"features": X_test, "labels": y_test}}

data_for_train = BinaryDataset(dataset)

criterion = nn.BCELoss()

R = 10.
eps = 0.1
eta = 100

maxiter = 4000

all_losses = []

for BATCH_SIZE in BATCH_SIZEs:
    A_0, b_0 = get_init_polytope_square(n, R)
    torch.manual_seed(0)
    net = LogisticRegression(55, 1)
    W_0 = net.fc.weight.detach().numpy()

    data_train = DataLoader(dataset=data_for_train, batch_size=BATCH_SIZE, shuffle=False)

    Ws = vaidya_for_logreg(A_0, b_0, W_0.T, eps, eta, oracle, get_loss,
                           net, data_train, criterion, stepsize=0.18, maxiter=maxiter)
    start = time.time()
    losses = []
    net.eval()
    with torch.no_grad():
        for W in Ws:
            net.fc.weight = torch.nn.Parameter(torch.from_numpy(W.T.astype(np.float32)))

            pred = net(X_test).flatten()
            cur_loss = criterion(pred, y_test).item()

            losses.append(cur_loss)
    print(f"Evaluation took {time.time() - start:.1f} s")
    with open(f'Vaidya_batch_{BATCH_SIZE}.pickle', 'wb') as handle:
        pickle.dump(losses, handle)
    all_losses.append(losses)

plt.xlabel('Iteration')
plt.ylabel('Loss')
for i, losses in enumerate(all_losses):
    plt.plot(losses, label=f"batch size {BATCH_SIZEs[i]}")
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(40))
plt.grid(True, linestyle='-', color='0.75')
plt.legend()
plt.ylim(top=1)
plt.xlim(left=0)

plt.savefig('vaidya_loss5.png')
