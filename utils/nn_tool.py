import random

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def fix_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_conv(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)