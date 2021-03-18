import torch

# 2 含隐藏状态的循环神经⽹络
# 连接矩阵乘法等价
X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)
print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))

print(torch.matmul(torch.cat((X, H), dim=1), torch.cat((W_xh, W_hh), dim=0)))
#3 应⽤：基于字符级循环神经⽹络的语⾔模型

