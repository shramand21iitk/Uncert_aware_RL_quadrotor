import math
import torch
import torch.nn as nn
import torch.nn.functional as F
_LOG_2PI = math.log(2*math.pi)

class PNN(nn.Module):
    def __init__(self, input_dim=23, output_dim=19):
        super().__init__()
        self.input_dim = input_dim
        self.neural_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU())
        self.mean = nn.Linear(128, output_dim)
        self.logvar = nn.Linear(128, output_dim)
        self.max_logvar = nn.Parameter(torch.ones(1, output_dim)*(1.5))
        self.min_logvar = nn.Parameter(torch.ones(1, output_dim)*(-9))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input):
        x = self.neural_net(input)
        mean = self.mean(x)
        logvar = self.logvar(x)
        logvar = self.max_logvar-F.softplus(self.max_logvar-logvar)
        logvar = self.min_logvar+F.softplus(logvar-self.min_logvar)
        variance = torch.exp(logvar)
        return mean, variance
    
    def gauss_beta_log_likelihood_loss(self, target, pred_mean, pred_var, beta):
        pred_var = pred_var + 1e-6
        loss = -0.5 * (((target - pred_mean)**2 / pred_var) + torch.log(pred_var) + _LOG_2PI)
        weight = (pred_var.detach() + 1e-6) ** beta
        batch_loss = -torch.sum(loss * weight, axis=-1)
        return torch.mean(batch_loss)