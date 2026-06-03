import torch
import torch.nn as nn
import torch.nn.functional as F


class Medium(nn.Module):
    def __init__(self, input_size, output_size):
        super(Medium, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MediumStochastic(nn.Module):
    def __init__(self, input_size, output_size):
        super(MediumStochastic, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 10)
        self.mu = nn.Linear(10, output_size)
        self.log_sigma = nn.Linear(10, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        return mu

    def forward_mu_sigma(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        sigma = F.softplus(log_sigma) + 1e-6
        return mu, sigma
