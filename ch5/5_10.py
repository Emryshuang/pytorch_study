import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
本节我们介绍批量归⼀化（batch normalization）层，它能让较深的神经⽹络的训练变得更加容易
。在3.16节（实战Kaggle⽐赛：预测房价）⾥，我们对输⼊数据做了标准化处理：处理后的任意⼀
个特征在数据集中所有样本上的均值为0、标准差为1。标准化处理输⼊数据使各个特征的分布相近：这
往往更容易训练出有效的模型。
通常来说，数据标准化预处理对于浅层模型就⾜够有效了。随着模型训练的进⾏，当每层中参数更新
时，靠近输出层的输出较难出现剧烈变化。但对深层神经⽹络来说，即使输⼊数据已做标准化，训练中
模型参数的更新依然很容易造成靠近输出层输出的剧烈变化。这种计算数值的不稳定性通常令我们难以
训练出有效的深度模型。
批量归⼀化的提出正是为了应对深度模型训练的挑战。在模型训练时，批量归⼀化利⽤⼩批量上的均值
和标准差，不断调整神经⽹络中间输出，从⽽使整个神经⽹络在各层的中间输出的数值更稳定。批量归
⼀化和下⼀节将要介绍的残差⽹络为训练和设计深度模型提供了两类重要思路。
'''


def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps,
               momentum):
    if not is_training:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=0,
                          keepdim=True).mean(dim=2,
                                             keepdim=True).mean(dim=3,
                                                                keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(
                dim=2, keepdim=True).mean(dim=3, keepdim=True)

        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(device)
            self.moving_var = self.moving_var.to(device)
        Y, self.moving_mean, self.moving_var = batch_norm(self.training,
                                                          X,
                                                          self.gamma,
                                                          self.beta,
                                                          self.moving_mean,
                                                          self.moving_var,
                                                          eps=1e-5,
                                                          momentum=0.9)
        return Y


net = nn.Sequential(nn.Conv2d(1, 6, 5), nn.Sigmoid(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(6, 16, 5), BatchNorm(16, num_dims=4),
                    nn.Sigmoid(), nn.MaxPool2d(2, 2), d2l.FlattenLayer(),
                    nn.Linear(16 * 4 * 4, 120), BatchNorm(120, num_dims=2),
                    nn.Sigmoid(), nn.Linear(120, 84), BatchNorm(84,
                                                                num_dims=2),
                    nn.Sigmoid(), nn.Linear(84, 10))

batch_size = 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

lr, num_epochs = 0.001, 5
optimizer = optim.Adam(net.parameters(), lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

# print(net[1].gamma.view((-1,)), net[1].beta.view((-1,)))

net = nn.Sequential(nn.Conv2d(1, 6, 5), BatchNorm(6, num_dims=4),
                    nn.Sigmoid(),
                    nn.MaxPool2d(2, 2),  # kernel_size, stride
                    nn.Conv2d(6, 16, 5),
                    BatchNorm(16, num_dims=4),
                    nn.Sigmoid(),
                    nn.MaxPool2d(2, 2),
                    d2l.FlattenLayer(),
                    nn.Linear(16 * 4 * 4, 120),
                    BatchNorm(120, num_dims=2),
                    nn.Sigmoid(),
                    nn.Linear(120, 84),
                    BatchNorm(84, num_dims=2),
                    nn.Sigmoid(),
                    nn.Linear(84, 10)
                    )

batch_size = 64
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

lr, num_epochs = 0.001, 5
optimizer = optim.Adam(net.parameters(), lr=lr)

d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
