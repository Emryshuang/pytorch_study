import math
from IPython import display
from matplotlib import pyplot as plt
import time
import torchvision
import random
import torch
import sys
from torch import nn
import torch.nn.functional as F
import zipfile
import numpy as np


class D2l(object):
    pass


def use_svg_isplay():
    display.set_matplotlib_formats('svg')


# 3.2.2
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_isplay()
    plt.rcParams['figure.figsize'] = figsize


# 3.2.2
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


# 3.2.4
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 3.2.5定义损失函数
def squared_loss(y_hat, y):
    # 此处返回的是向量，pytorch里的MSELoss并没有除以2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 3.2.6定义优化函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 3.5.1获取数据集
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle',
                   'boot']
    return text_labels


# 3.5.1获取数据集
def show_fashion_mnist(images, labels):
    use_svg_isplay()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 3.5.1获取数据集
def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


# 3.6.6计算分类准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 3.6.7 训练模型

def train_ch3(net, train_iter, test_iter, loss, num_epchos, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epchos):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f,test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


# 3.7.2 定义和初始化模型
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


# 3.11.4.2 定义、训练和测试模型
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


# 3.13.2 从零开始实现drop
def evalute_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train()
        else:
            if ('is_training' in net.__code__.co_varnames):
                acc_sum += (net(X, is_trainint=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 5.1.1 ⼆维互相关运算
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# 5.5.2 评估准确度
def evaluate_accuracy(
        data_iter,
        net,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(
                    dim=1) == y.to(device)).float().sum().cpu().item()
            else:
                if ('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(
                        dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(
                        dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


# 5.5.2  ch5训练
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device,
              num_epochs):
    net = net.to(device)
    print("trainint on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               test_acc, time.time() - start))


# 5.6.3 读取数据
def load_data_fashion_mnist(batch_size,
                            resize=None,
                            root='~/Datasets/FashionMNIST'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root,
                                                    train=True,
                                                    download=False,
                                                    transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root,
                                                   train=False,
                                                   download=False,
                                                   transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size,
                                             shuffle=True
                                             )
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size,
                                            shuffle=False
                                            )

    return train_iter, test_iter


# 5.8.2全局平均池化层
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


# 6.3.1 读取数据集  加载周杰伦歌词数据集
def load_data_jay_lyrics():
    with zipfile.ZipFile('../data/ch6/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


# 6.3.3.1 随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i:i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


# 6.3.3.2 相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0:batch_size * batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i:i + num_steps]
        Y = indices[:, i + 1:i + num_steps + 1]
        yield X, Y


# 6.4.1 ONE-HOT向量
def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


# 6.4.1 ONE-HOT向量
def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


# 6.4.4 定义预测函数
def predict_run(prefix, num_chars, run, params, init_run_state, num_hiddens, vocab_size, device, idx_to_char,
                char_to_idx):
    state = init_run_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        (Y, state) = run(X, state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


# 6.4.5 裁剪梯度
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


# 6.4.7 定义模型训练函数
def train_and_predict_rnn(rnn, get_params, init_run_state, num_hiddens, vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps, lr, clipping_theta, batch_size,
                          pred_period, pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # 使用相邻采样，在epoch开始时初始化隐藏状态
        if not is_random_iter:
            state = init_run_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            # 使用随机采样，在每个小批量更新前初始化隐藏状态
            if is_random_iter:
                state = init_run_state(batch_size, num_hiddens, device)
            else:
                for s in state:
                    s.detach_()

            inputs = to_onehot(X, vocab_size)

            (outputs, state) = rnn(inputs, state, params)

            outputs = torch.cat(outputs, dim=0)

            y = torch.transpose(Y, 0, 1).contiguous().view(-1)

            l = loss(outputs, y.long())

            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            # 裁剪梯度
            grad_clipping(params, clipping_theta, device)
            # 小批量下降，梯度已经裁剪，不做评价，batch_size为1
            sgd(params, lr, 1)

            l_sum += l.item() * y.shape[0]

            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            # print('epoch %d\tn = %d' % (epoch, n))
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_run(prefix, pred_len, rnn, params, init_run_state, num_hiddens, vocab_size, device,
                                        idx_to_char, char_to_idx))


# 6.5.1 定义模型
class RNNModel(nn.Module):

    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, self.vocab_size)
        self.state = None

    def forward(self, inputs, state):
        X = to_onehot(inputs, self.vocab_size)
        Y, self.state = self.rnn(torch.stack(X), state)

        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


# 6.5.2 训练模型

def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char, char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


# 6.5.2 训练模型
def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len,
                                  prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if state is not None:
                if isinstance(state, tuple):
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            (output, state) = model(X, state)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()

            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(prefix, pred_len, model, vocab_size, device, idx_to_char, char_to_idx))


# 7.2.3 多维梯度下降
def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results


# 7.2.3 多维梯度下降
def show_trace_2d(f, results):
    plt.plot(*zip(*results), '-o', color='#ffff00')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#9999FF')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


# 7.3.1 读取数据
def get_data_ch7():
    data = np.genfromtxt('../data/ch7/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1],
                        dtype=torch.float32), torch.tensor(data[:1500, -1],
                                                           dtype=torch.float32)

# 7.3.1 读取数据
def train_ch7(optimizer_fn,
              states,
              hyperparqms,
              features,
              lables,
              batch_size=10,
              num_epochs=2):
    net, loss = linreg, squared_loss

    w = torch.nn.Parameter(torch.tensor(np.random.normal(
        0, 0.01, size=(features.shape[1], 1)),
        dtype=torch.float32),
        requires_grad=True)
    b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32),
                           requires_grad=True)

    def eval_loss():
        return loss(net(features, w, b), lables).mean().item()

    ls = [eval_loss()]

    data_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        features, lables),
        batch_size,
        shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            l.backward()
            optimizer_fn([w, b], states, hyperparqms)
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print('loss: %f, %f sec per epoch ' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


# 7.3.3 简洁实现
def train_pytorch_ch7(optimizer_fn,
                      optimizer_hyperparams,
                      features,
                      labels=None,
                      batch_size=10,
                      num_epochs=2):
    if labels == None:
        _, labels = get_data_ch7()
    net = nn.Sequential(nn.Linear(features.shape[-1], 1))
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        features, labels),
        batch_size,
        shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X).view(-1), y) / 2

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print('loss: %f, %f sec per epoch ' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
