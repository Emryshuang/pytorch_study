import torch
import torch.nn as nn
import numpy as np
import d2l_pytorch as d2l


# 2 从零开始实现
def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob

    if keep_prob == 1:
        return torch.zeros_like(X)
    mask = (torch.randn(X.shape) < keep_prob).float()

    return mask * X / keep_prob

'''
X = torch.arange(16).view(2, 8)
print(dropout(X, 0))
print(dropout(X, 0.5))
print(dropout(X, 1.0))
'''

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
