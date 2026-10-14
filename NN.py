import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, input_dim=23, output_dim=19):
        super().__init__()
        self.input_dim = input_dim
        self.neural_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU()
        )
        self.out = nn.Linear(128, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input):
        x = self.neural_net(input)
        out = self.out(x)
        return out