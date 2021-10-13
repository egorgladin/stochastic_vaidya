import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, result_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, result_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))
