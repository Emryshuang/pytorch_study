import torch
from torch import nn
import d2l_pytorch.d2l as d2l


# 1 多输⼊通道

# 输入数据含多个通道时，需要构造输入通道数与输入的数据的通道数相同的卷积核
def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道维）分别计算再计算
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res


X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(corr2d_multi_in(X, K))


# 2 多输出通道

# 使用相同的输入，使用不同组的卷积核张量创造不同的输出
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K])


K = torch.stack([K, K + 1, K + 2])
print(K.shape)

print(corr2d_multi_in_out(X, K))


# 3 卷积层

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    # mm 全连接层的矩阵乘法
    Y = torch.mm(K, X)
    return Y.view(c_o, h, w)


X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print((Y1 - Y2).norm().item() < 1e-6)
print((Y1 - Y2).norm().item())
