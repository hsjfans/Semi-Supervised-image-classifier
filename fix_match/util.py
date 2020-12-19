import torch.nn as nn


def clones(layer, num):
    return nn.ModuleList([layer for _ in range(num)])
