import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    # You should build your model with at least 2 layers using tanh activation in between
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        return self.fc3(x)
        