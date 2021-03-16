import torch
from torch import nn
import d2l_pytorch.d2l as d2l

# 1 ⼆维互相关运算
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(d2l.corr2d(X, K))


# 2 ⼆维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return d2l.corr2d(x, self.weight) + self.bias


# 3 图像中物体边缘检测
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1, -1]])

Y = d2l.corr2d(X, K)
print(Y)
# 4 通过数据学习核数组
conv2d = Conv2D(kernel_size=(1, 2))

step = 40
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)

    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % ((i + 1), l.item()))

print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)
