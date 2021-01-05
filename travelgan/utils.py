import torch
import torch.nn as nn

def initialize_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def spectral_normalization(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m = nn.utils.spectral_norm(m)