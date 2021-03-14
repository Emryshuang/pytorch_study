import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

# 1 生成数据集
num_imputs = 2
num_examples = 1000
true_w = [2, -3, 4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_imputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
# 2 读取数据
batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break


# 3定义模型

class LinerNet(nn.Module):
    def __init__(self, n_feature):
        super(LinerNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


'''

net = LinerNet(num_imputs)
print(net)

net = nn.Sequential(
    nn.Linear(num_imputs, 1)
)

net = nn.Sequential()
net.add_module('linear',nn.Linear(num_imputs, 1))

from collections import OrderedDict
net = nn.Sequential(OrderedDict([
    ('linear',nn.Linear(num_imputs,1))
]))

print(net)
print(net[0])

'''

net = nn.Sequential(
    nn.Linear(num_imputs, 1)
)

for param in net.parameters():
    print(param)

# 4初始化模型参数
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)
# net[0].bias.data.fill_(0)

# 5定义损失函数
loss = nn.MSELoss()

# 6定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)
'''
为子网络设置不同的学习率
optimizer = optim.SGD(
    [
        {'params': net.subnet1.parameters()},
        {'params': net.subnet2.parameters(), 'lr': 0.01}
    ], lr=0.03
)
'''

for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1

num_epochs = 10
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss %f' % (epoch, l.item()))

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
