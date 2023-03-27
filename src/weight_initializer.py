import math
import torch
from torch import nn


def xavier_uniform(n_in, n_out, device):

    bound = math.sqrt(3.0) * math.sqrt(2.0 / float(n_in + n_out))
    return nn.Parameter(torch.rand((n_in, n_out)).to(device)*(2*bound)-bound)