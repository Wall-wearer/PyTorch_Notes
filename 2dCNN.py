# This is a file to do things like 2D-CNN(二维卷积层, convolutional neural network)，其实就是库里面的东西，这里再给他解释一下

import torch
from torch import nn


# 卷积运算（二维互相关）
# 这里是进行卷积运算的步骤
def corr2d(X, K):
    h, w = K.shape      # h, w获取了K的行列数K.shape=[行，列]，这里的可以是卷积核convolutional kernel
    X, K = X.float(), K.float()     # 把X，K转换成浮点形式
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # 根据X的行列，创建一个Y。X.shape为行列式，用[行，列]，shape[0]=为行
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
            # 从i行->i+h行，j列->j+w列的行列式。
    return Y


# 二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):    # Size of the convolutional kernel.
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))    # Learnable kernel weights
        self.bias = nn.Parameter(torch.randn(1))    # Learnable bias term

    def forward(self, x):   # forward method
        return corr2d(x, self.weight) + self.bias


# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:]) # 排除不关心的前两维:批量和通道


# 注意这里是两侧分别填充1⾏或列，所以在两侧一共填充2⾏或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,padding=1)


# 开始创建新数组
X = torch.rand(8, 8)
comp_conv2d(conv2d, X).shape

print(comp_conv2d(conv2d, X).shape)


# 卷积核尺寸3x5，使用填充padding，行的填充为0，宽的填充为1。步幅stride为3，4
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape

