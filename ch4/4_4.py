import torch
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


layer = CenteredLayer()
print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
y = net(torch.rand(4, 8))
print(y.mean().item())


# 2 含模型参数的⾃定义层
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.param = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.param.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x


net = MyDense()
print(net)


class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.rand(4, 1))
        })
        self.params.update({
            'linear3': nn.Parameter(torch.randn(4, 2))
        })

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])


net = MyDictDense()
print(net)

x=torch.ones(1,4)
print(net(x,'linear1'))
print(net(x,'linear2'))
print(net(x,'linear3'))

net = nn.Sequential(
    MyDictDense(),
    #MyListDensen()
)
print(net)
print(net(x))