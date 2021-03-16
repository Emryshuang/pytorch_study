import torch
from torch import nn


# 池化（pooling）层，它的提出是为了缓解卷积层对位置的过度敏感性。
# 1 ⼆维最⼤池化层和平均池化层

def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()

    return Y


# 最大池化层
X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(pool2d(X, (2, 2)))
# 平均池化层
print(pool2d(X, (2, 2), 'avg'))

# 2 填充和步幅

X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
# 3默认为(3, 3)
pool2d = nn.MaxPool2d(3)
print(pool2d(X))
# 填充1 步幅2
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
print(pool2d(X))

# 3 多通道
X = torch.cat((X, X + 1), dim=1)
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
