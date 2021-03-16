import time
import torch
from torch import nn, optim
import d2l_pytorch.d2l as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
AlexNet在LeNet的基础上增加了3个卷积层。但AlexNet作者对它们的卷积窗⼝、输出通道数和构造顺
序均做了⼤量的调整。虽然AlexNet指明了深度卷积神经⽹络可以取得出⾊的结果，但并没有提供简单
的规则以指导后来的研究者如何设计新的⽹络。

VGG块的组成规律是：连续使⽤数个相同的填充为1、窗⼝形状为3X3的卷积层后接上⼀个步幅为2、
窗⼝形状为2X2的最⼤池化层。卷积层保持输⼊的⾼和宽不变，⽽池化层则对其减半。我们使
⽤ vgg_block 函数来实现这个基础的VGG块，它可以指定卷积层的数量和输⼊输出通道数。
Visual Geometry Group 

'''


# 1 vgg块
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


'''
2 VGG网络
构造⼀个VGG⽹络。它有5个卷积块，前2块使⽤单卷积层，⽽后3块使⽤双卷积层。第⼀块的
输⼊输出通道分别是1（因为下⾯要使⽤的Fashion-MNIST数据的通道数为1）和64，之后每次对输出通
道数翻倍，直到变为512。因为这个⽹络使⽤了8个卷积层和3个全连接层，所以经常被称为VGG-11。
'''

conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512,
                                                                      512))
# 经过5个vgg_block, 宽⾼会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7
fc_hidden_units = 4096


def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()

    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i + 1),
                       vgg_block(num_convs, in_channels, out_channels))
    net.add_module(
        "fc",
        nn.Sequential(d2l.FlattenLayer(),
                      nn.Linear(fc_features, fc_hidden_units), nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(fc_hidden_units, fc_hidden_units), nn.ReLU(),
                      nn.Dropout(0.5), nn.Linear(fc_hidden_units, 10)))

    return net


net = vgg(conv_arch, fc_features, fc_hidden_units)
X = torch.rand(1, 1, 224, 224)

for name, blk in net.named_children():
    X = blk(X)
    print(name, ' output shapeL', X.shape)

# 3 获取数据并训练

ratio = 8
small_conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio),
                   (2, 128 // ratio, 256 // ratio),
                   (2, 256 // ratio, 512 // ratio),
                   (2, 512 // ratio, 512 // ratio)]

print(small_conv_arch)

net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
print(net)

batch_size = 32
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.001, 5
optimizer = optim.Adam(net.parameters(), lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
