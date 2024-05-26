import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.network.apply(self._init_weights)

    def forward(self, x):
        return self.network(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# Loss rate rapidly decreasing observed, added another hidden layer
# Restructured module to be dynamically resizable to run grid search


