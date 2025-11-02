import torch
import torch.nn as nn

class HiddenLayer(nn.Module):
    """A reusable hidden layer (Linear + ReLU)."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
    def forward(self, x):
        return self.activation(self.linear(x))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=64, hidden_layers=2):
        super().__init__()
        assert hidden_layers >= 1, "Actor must have at least one hidden layer"
        layers = [HiddenLayer(state_dim, hidden_dim)]
        for _ in range(hidden_layers - 1):
            layers.append(HiddenLayer(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.max_action = max_action
    def forward(self, state):
        return self.net(state) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, hidden_layers=2):
        super().__init__()
        assert hidden_layers >= 1, "Critic must have at least one hidden layer"
        input_dim = state_dim + action_dim
        layers = [HiddenLayer(input_dim, hidden_dim)]
        for _ in range(hidden_layers - 1):
            layers.append(HiddenLayer(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, state, action):
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = torch.cat([state, action], dim=1)
        return self.net(x)
