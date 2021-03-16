import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
对神经⽹络模型添加新的层，充分训练后的模型是否只可能更有效地降低训练
误差？理论上，原模型解的空间只是新模型解的空间的⼦空间。也就是说，如果我们能将新添加的层训
练成恒等映射f(x)=x,新模型和原模型将同样有效。由于新模型可能得出更优的解来拟合训练数据
集，因此添加层似乎更容易降低训练误差。然⽽在实践中，添加过多的层后训练误差往往不降反升。即
使利⽤批量归⼀化带来的数值稳定性使训练深层模型更加容易，该问题仍然存在。针对这⼀问题，何恺
明等⼈提出了残差⽹络（ResNet）。它在2015年的ImageNet图像识别挑战赛夺魁，并深刻影响了
后来的深度神经⽹络的设计。

ResNet沿⽤了VGG全3X3卷积层的设计。残差块⾥⾸先有2个有相同输出通道数的3X3卷积层。每
个卷积层后接⼀个批量归⼀化层和ReLU激活函数。然后我们将输⼊跳过这两个卷积运算后直接加在最后
的ReLU激活函数前。这样的设计要求两个卷积层的输出与输⼊形状⼀样，从⽽可以相加。如果想改变通
道数，就需要引⼊⼀个额外的1X1卷积层来将输⼊变换成需要的形状后再做相加运算。
'''


# 1 残差块


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1,
                               stride=stride)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


blk = Residual(3, 3)
X = torch.rand((4, 3, 6, 6))
blk(X).shape

blk = Residual(3, 6, use_1x1conv=True, stride=2)
blk(X).shape

# 2  RESNET模型


net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(in_channels, out_channels, use_1x1conv=True,
                         stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))

net.add_module("global_avg_pool", d2l.GlobalAvgPool2d())
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10)))

X = torch.rand((1, 1, 224, 224))

for name, layer in net.named_children():
    X = layer(X)
    print(name, 'output shape:\t', X.shape)

print(net)

# 3 获取数据和训练模型


batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=96)

lr, num_epochs = 0.001, 5
optimizer = optim.Adam(net.parameters(), lr=lr)

d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
