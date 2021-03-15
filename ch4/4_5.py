import torch
from torch import nn

# 1 读写 TENSOR
x = torch.ones(3)
PATH = '../data/ch4/'
torch.save(x, PATH + 'x.pt')

x2 = torch.load(PATH + 'x.pt')
print(x2)

y = torch.zeros(4)
torch.save([x, y], PATH + 'xy.pt')
xy_list = torch.load(PATH + 'xy.pt')
print(xy_list)

torch.save({'x': x, 'y': y}, PATH + 'xy_dict.pt')
xy = torch.load(PATH + 'xy_dict.pt')
print(xy)


# 2 读写模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


net = MLP()
print(net.state_dict())
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())

'''
2.1. 保存和加载 state_dict (推荐⽅式)
保存
torch.save(model.state(),PATH)
加载
modol = TheModelClass(*arg,**kwargs)
model.load_state_dic(torch.load(PATH))
2.2. 保存和加载整个模型
保存
torch.save(model,PATH)
加载
model = torch.load(PATH)

'''

X = torch.randn(2, 3)
Y = net(X)

torch.save(net.state_dict(), PATH + 'net.pt')

net2 = MLP()
net2.load_state_dict(torch.load(PATH + 'net.pt'))
Y2 = net2(X)
print(Y)
print(Y2)
print(Y == Y2)

# 2.2. 保存和加载整个模型
