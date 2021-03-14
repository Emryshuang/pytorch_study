from IPython import display
from matplotlib import pyplot as plt
import d2l_pytorch
import torchvision
import random
import torch
import sys

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

