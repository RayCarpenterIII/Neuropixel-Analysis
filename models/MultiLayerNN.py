import torch
import torch.nn as nn
import numpy as np

class MLNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MLNNModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            hidden_dim //= 2
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        batch_size, num_timesteps, num_nodes, num_features = x.size()
        print(f'shape of x = {np.shape(x)}')
        x = x.view(batch_size, num_timesteps, -1)  # Flatten the node and feature dimensions
        x = x.view(batch_size, -1)  # Flatten the timestep and flattened node-feature dimensions
        out = self.layers(x)
        return out