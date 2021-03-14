import torch
from torch import nn
from torch.nn import init
import numpy as np
import d2l_pytorch
from collections import OrderedDict

# 1获取和读取数据
batch_size = 256
train_iter, test_iter = d2l_pytorch.load_data_fashion_mnist(batch_size)
# 定义和初始化模型
num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y


net = LinearNet(num_inputs, num_outputs)


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


net = nn.Sequential(
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 3 SOFTMAX和交叉熵损失函数
loss = nn.CrossEntropyLoss()
# 4 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# 5 训练模型
num_epochs = 5
d2l_pytorch.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
