import torch
from torch import nn
import time

# 1 计算设备
print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.cuda.current_device())

print(torch.cuda.get_device_name())

# 2 TENSOR 的GPU计算

x = torch.tensor([1, 2, 3])
print(x)
start = time.time()
x = x.cuda(0)  # cuda(0)，cuda()等价
print(time.time() - start)
print(x)
print(time.time() - start)
print(x.device)

'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1,2,3],device=device)
x = torch.tensor([1,2,3]).to(device)
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1, 2, 3]).to(device)

print(x)
y = x ** 2
print(y)

# 3 模型的GPU计算
net = nn.Linear(3, 1)
print(list(net.parameters())[0].device)
net.cuda()
print(list(net.parameters())[0].device)
x = torch.rand(2, 3).cuda()
print(net(x))
