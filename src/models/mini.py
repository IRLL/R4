"""
small model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Mini(nn.Module):
    def __init__(self, input_size, output_size):
        super(Mini, self).__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MiniMPH:
    """
    model for metropolis hastings.
    """

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.l1 = np.random.rand(5, input_size)
        self.l2 = np.random.rand(output_size, 5)
        self.bias1 = np.random.rand(5)
        self.bias2 = np.random.rand(output_size)

    def forward(self, x):
        x = np.dot(x, self.l1.T) + self.bias1
        x = np.maximum(x, 0)
        x = np.dot(x, self.l2.T) + self.bias2
        x = np.tanh(x)
        return x

    def get_weights(self):
        return np.concatenate([self.l1.flatten(), self.l2.flatten(), self.bias1, self.bias2])

    def set_weights(self, weights):
        self.l1 = weights[: 5 * self.input_size].reshape(5, self.input_size)
        self.l2 = weights[5 * self.input_size : 5 * self.input_size + self.output_size * 5].reshape(
            self.output_size, 5
        )
        self.bias1 = weights[5 * self.input_size + self.output_size * 5 : 5 * self.input_size + self.output_size * 5 + 5]
        self.bias2 = weights[5 * self.input_size + self.output_size * 5 + 5 :]

    def convert_to_torch(self):
        model = Mini(self.input_size, self.output_size)
        model.fc1.weight.data = torch.tensor(self.l1)
        model.fc1.bias.data = torch.tensor(self.bias1)
        model.fc2.weight.data = torch.tensor(self.l2)
        model.fc2.bias.data = torch.tensor(self.bias2)
        model = model.float()
        return model
