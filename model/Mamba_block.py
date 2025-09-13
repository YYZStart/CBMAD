import torch
import torch.nn as nn
from mamba_ssm import Mamba


class Mamba_block(nn.Module):
    def __init__(self, d_model, d_state, d_conv=4, expand=2, rms_norm=True):
        super(Mamba_block, self).__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        if rms_norm:

            self.mamba = Residual_Mamba_Block(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        else:

            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):

        x = self.mamba(x)

        return x


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class Residual_Mamba_Block(nn.Module):
    def __init__(self, d_model, d_state, d_conv=4, expand=2):
        super(Residual_Mamba_Block, self).__init__()

        self.nb_feature = d_model
        self.state_size = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = RMSNorm(d_model, 1e-5)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

