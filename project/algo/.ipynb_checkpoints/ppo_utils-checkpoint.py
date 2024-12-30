import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
import  torch

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, env, hidden_size=32):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
       
        self.fc1 = torch.nn.Linear(state_space, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.mean_layer = torch.nn.Linear(hidden_size, action_space)
        self.std_layer = torch.nn.Linear(hidden_size, action_space)
        self.fc_value = torch.nn.Linear(hidden_size, 1)

        self.init_weights()
        self.actor_logstd = torch.nn.Parameter(torch.zeros(action_space))
        
    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 1e-1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_mean = self.mean_layer(x)
        action_std = torch.exp(self.actor_logstd)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        state_value = self.fc_value(x)
        
        return action_dist, state_value
    def set_logstd_ratio(self, ratio):
            self.actor_logstd.data = torch.clamp(self.actor_logstd.data*ratio, min=-2.0, max=0.5)