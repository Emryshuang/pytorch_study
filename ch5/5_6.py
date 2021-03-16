import torch
from torch import nn, optim
import torchvision
import d2l_pytorch.d2l as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
1 学习特征表示

认为特征本身也应该由学习得来。他们还相信，为了表征⾜够复杂
的输⼊，特征本身应该分级表示。持这⼀想法的研究者相信，多层神经⽹络可能可以学得数据的多级表
征，并逐级表示越来越抽象的概念或模式。在多层神经⽹络中，图像的第⼀级的表示可以是在特定的位置和⻆度是否出现边缘；⽽第
⼆级的表示说不定能够将这些边缘组合出有趣的模式，如花纹；在第三级的表示中，也许上⼀级的花纹
能进⼀步汇合成对应物体特定部位的模式。这样逐级表示下去，最终，模型能够较容易根据最后⼀级的
表示完成分类任务。需要强调的是，输⼊的逐级表示由多层模型中的参数决定，⽽这些参数都是学出来
的。
'''


# 2  ALEXNET

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 96, 11, 4), nn.ReLU(),
                                  nn.MaxPool2d(3, 2),
                                  # 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
                                  nn.Conv2d(96, 256, 5, 1, 2), nn.ReLU(),
                                  nn.MaxPool2d(3, 2),
                                  # 连续3个卷积层，且使⽤更⼩的卷积窗⼝。除了最后的卷积层外，进⼀步增⼤了输出通道数。
                                  # 前两个卷积层后不使⽤池化层来减⼩输⼊的⾼和宽
                                  nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(),
                                  nn.Conv2d(384, 384, 3, 1, 1), nn.ReLU(),
                                  nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(),
                                  nn.MaxPool2d(3, 2))
        # 这⾥全连接层的输出个数⽐LeNet中的⼤数倍。使⽤丢弃层来缓解过拟合
        self.fc = nn.Sequential(nn.Linear(256 * 5 * 5, 4096), nn.ReLU(),
                                nn.Dropout(0.5), nn.Linear(4096, 4096),
                                nn.ReLU(), nn.Dropout(0.5),
                                nn.Linear(4096, 10))

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


net = AlexNet()
print(net)

# 3 读取数据

'''
def load_data_fashion_mnist(batch_size,
                            resize=None,
                            root='~/Datasets/FashionMNIST'):
'''

# 4 训练


batch_size = 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.001, 5
optimizer = optim.Adam(net.parameters(), lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
