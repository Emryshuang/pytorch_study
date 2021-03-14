import torch
from torch import nn
from torch.nn import init
import numpy as np
import d2l_pytorch as d2l

# 1 定义模型
num_inputs, num_outputs, num_hidden = 784, 10, 256

net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hidden),
    nn.ReLU(),
    nn.Linear(num_hidden, num_outputs),
)

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)
# 2 读取数据并训练模型

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs=5
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)