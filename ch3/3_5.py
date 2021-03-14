import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import d2l_pytorch as d2l
import sys

mnist_train = torchvision.datasets.FashionMNIST(
    root='~/Datasets/FashionMNIST',
    train=True,
    download=False,
    transform=transforms.ToTensor()
)
mnist_test = torchvision.datasets.FashionMNIST(
    root='~/Datasets/FashionMNIST',
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

print(type(mnist_train), type(mnist_test))
print(len(mnist_train), len(mnist_test))
feature, label = mnist_train[0]
print(feature.shape, label)
'''
torch.Size([1, 28, 28]) 9
通道数 高 宽
'''


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle',
                   'boot']
    return text_labels


def show_fashion_mnist(images, labels):
    d2l.use_svg_isplay()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 2读取小批量
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
