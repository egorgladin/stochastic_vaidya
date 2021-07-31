import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self, input_dim, result_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, result_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def forward(self, X):
        return self.sigmoid(self.fc(X))
    
    def get_loss(self, X, y):
        self.zero_grad()
        output = self.forward(X)
        return self.loss_fn(output, y.unsqueeze(dim=1))
        
    def oracle(self, X, y):
        loss = self.get_loss(X, y)
        loss.backward()
        return self.fc.weight.grad.detach().numpy()
    
    def update_weights(self, W):
        with torch.no_grad():
            self.fc.weight = W