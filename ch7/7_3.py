import numpy as np
import time
import torch
from torch import optim, nn
import d2l_pytorch.d2l as d2l

# 1 读取数据


features, labels = d2l.get_data_ch7()
print(features.shape)


# 2 从零开始实现
def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data


def train_sgd(lr, batch_size, num_epochs=2):
    d2l.train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)


train_sgd(1, 1500, 6)

train_sgd(0.05, 10)

train_sgd(0.05, 1)

# 3 简介实现

d2l.train_pytorch_ch7(optim.SGD, {"lr": 0.05}, features, labels, 10)

d2l.train_pytorch_ch7(optim.SGD, {"lr": 0.05}, features, labels, 1500)
