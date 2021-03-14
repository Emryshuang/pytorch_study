import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# 1
num_imput = 2
num_examples = 1000
true_w = [2, -3, 4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_imput)))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size()))

print(features.size())
print(labels.size())
print(features[0])
print(labels[0])
print(features[0], labels[0])


def use_svg_isplay():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_isplay()
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)


# plt.show()


# 2
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10
'''
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

'''
# 3
w = torch.tensor(np.random.normal(0, 0.01, (num_imput, 1)), dtype=torch.float64)
b = torch.zeros(1, dtype=torch.float64)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 4定义模型
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 5定义损失函数
def squared_loss(y_hat, y):
    # 此处返回的是向量，pytorch里的MSELoss并没有除以2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 6定义优化函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 7训练模型

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()

        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\t', w)
print(true_b, '\t', b)
