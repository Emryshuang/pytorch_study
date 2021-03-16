import time
import torch
from torch import nn, optim
import d2l_pytorch.d2l as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1 LENET模型

'''
卷积层块的输出形状为(批量⼤⼩, 通道, ⾼, 宽)。当卷积层块的输出传⼊全连接层块时，全连接层块会
将⼩批量中每个样本变平（flatten）。也就是说，全连接层的输⼊形状将变成⼆维，其中第⼀维是⼩批
量中的样本，第⼆维是每个样本变平后的向量表示，且向量⻓度为通道、⾼和宽的乘积。全连接层块含
3个全连接层。它们的输出个数分别是120、84和10，其中10为输出的类别个数。
'''


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 输入数据为28X28的图片
        self.conv = nn.Sequential(nn.Conv2d(1, 6, 5), nn.Sigmoid(),
                                  nn.MaxPool2d(2, 2), nn.Conv2d(6, 16, 5),
                                  nn.Sigmoid(), nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Linear(16 * 4 * 4, 120), nn.Sigmoid(),
                                nn.Linear(120, 84), nn.Sigmoid(),
                                nn.Linear(84, 10))

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


net = LeNet()
print(net)

'''
卷积层块⾥的基本单位是卷积层后接最⼤池化层：卷积层⽤来识别图像⾥的空间模式，如线条和物体局
部，之后的最⼤池化层则⽤来降低卷积层对位置的敏感性。卷积层块由两个这样的基本单位᯿复堆叠构
成。在卷积层块中，每个卷积层都使⽤5X5的窗⼝，并在输出上使⽤sigmoid激活函数。第⼀个卷积层
输出通道数为6，第⼆个卷积层输出通道数则增加到16。这是因为第⼆个卷积层⽐第⼀个卷积层的输⼊
的⾼和宽要⼩，所以增加输出通道使两个卷积层的参数尺⼨类似。卷积层块的两个最⼤池化层的窗⼝形
状均为2X2，且步幅为2。由于池化窗⼝与步幅形状相同，池化窗⼝在输⼊上每次滑动所覆盖的区域互
不᯿叠。
'''

# 2 获取数据和训练模型

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
