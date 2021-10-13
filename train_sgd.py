import json
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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


EPOCHS = 1
BATCH_SIZE = 25000
DATA_PATH = "../Downloads/covtype.libsvm.binary.scale"

A, y, m, n = prepare_data(DATA_PATH)
X, y = change_format(A, y, m)

net = LogisticRegression(55, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
dataset = {"train": {"features": X_train, "labels": y_train},
           "test": {"features": X_test, "labels": y_test}}

data_for_train = BinaryDataset(dataset)
data_for_test = BinaryDataset(dataset, mode="test")
data_train = DataLoader(dataset=data_for_train, batch_size=BATCH_SIZE, shuffle=False)
data_test = DataLoader(dataset=data_for_test, batch_size=BATCH_SIZE, shuffle=False)
criterion = nn.BCELoss()
optimizer = SGD(net.parameters(), lr=0.1, momentum=0.9)

loss_values = []

for epoch in range(EPOCHS):

    epoch_loss = 0

    for bidx, batch in enumerate(data_train):

        x_train, y_train = batch['inp'], batch['oup']
        loss, predictions = train(net, x_train, y_train, criterion)
        loss_values.append(float(loss))
        epoch_loss += loss

        logging.info('Step {} Loss : {}'.format((bidx + 1), loss))

    logging.info('Epoch {} Loss : {}'.format((epoch + 1), epoch_loss))

plt.title('BCE Loss for SGD')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.plot(range(1, len(loss_values) + 1), loss_values, 'r')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.grid(True, linestyle='-', color='0.75')
plt.savefig('sgd_loss.png')

with open("sgd_loss.json", 'w') as f:
    json.dump(loss_values, f, indent=2)
