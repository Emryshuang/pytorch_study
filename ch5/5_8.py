import torch
from torch import nn, optim
import d2l_pytorch.d2l as d2l
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
⽹络中的⽹络（NiN）。它提出了另外⼀个思路，即串联多个由卷积层
和“全连接”层构成的⼩⽹络来构建⼀个深层⽹络。

NiN块是NiN中的基础块。它由⼀个卷积层加两个充当全连接层的 卷积层串联⽽成。其中第
⼀个卷积层的超参数可以⾃⾏设置，⽽第⼆和第三个卷积层的超参数⼀般是固定的。
'''


# 1 NIN块
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )
    return blk


'''
NiN是在AlexNet问世不久后提出的。它们的卷积层设定有类似之处。NiN使⽤卷积窗⼝形状分别为
11X11、5X5和3X3的卷积层，相应的输出通道数也与AlexNet中的⼀致。每个NiN块后接⼀个步
幅为2、窗⼝形状为3X3的最⼤池化层。
'''


# 2 NIN模型

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


net = nn.Sequential(nin_block(1, 96, kernel_size=11, stride=4, padding=0),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=3, stride=2), nn.Dropout(0.5),
                    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
                    GlobalAvgPool2d(), d2l.FlattenLayer())

# 卷积层的输⼊和输出通常是四维数组（样本，通道，⾼，宽）
X = torch.rand(1, 1, 224, 224)
for name, blk, in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)

print(net)

# 3 获取数据和训练模型

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.002, 20
optimizer = optim.Adam(net.parameters(), lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device,num_epochs)
