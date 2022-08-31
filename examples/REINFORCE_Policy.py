import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
torch.manual_seed(0)

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Policy(nn.Module):
    def __init__(self, state_size=3, action_size=3, hidden_size=32):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

        self._center = torch.tensor([0.25,0.0,1.002], dtype=torch.float32, device=device)
        self._scale = torch.tensor([0.25,0.35,0.25], dtype=torch.float32, device=device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.tanh(x) * self._scale + self._center

    def get_action(self, state, quad, gripper):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_xyz = self.forward(state).cpu().detach().numpy().flatten()
        return np.concatenate((action_xyz, quad, np.array([gripper])))
