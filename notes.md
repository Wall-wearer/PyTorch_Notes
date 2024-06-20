# PyTorch

基础概念：张量(Tensor)，可以理解为矩阵，所有的机器学习的东西都存在数组里面。图像一般为4D张量。

'(batch_size, width, height, channel) = 4D'

## 索引操作
`a = x[:, 0]`(x是已经定义了的数组)

索引出来的数值和原数值有超链接，改一个会影响另一个。要是想不受影响，要用`copy`

## 纬度变化

`torch.view()`和`torch.reshape()`

注意，.view 来改变的只是观察数组的方式，如果改了这里的数据，还是会改两个。

z = x.view(-1, 8)
可以把原本 4* 4的数组分解成 2* 8的。这里可以自己算好几乘几，也可以直接输入-1，就让它自己算了。当然得整除。
他们似乎是共享内存的
要是想要不共享，就要用`reshape()`

## 广播机制(broadcasting)

用来解决计算2x3和3x2，这种结构形状不一样的数组。

```
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)
```

先创建1，3排列成横的，在创建1，4排列成竖的。然后两者相加，会成为一个2x3的。

## Autograd

自动求导，最后加上`requires_grad=True`触发。用来记录每一步。用来加速，和精确。`print(b.grad_fn)`

## 主要流程

1. 配置基本内容，比如设置尺寸，调用gpu等
2. 读入数据
3. 构建模型
4. 模型初始化
5. 损失评估
6. 训练和评估
7. 可视化

### 流程内详解

1. 基本配置。
   1. batch size：每次输入数量
   2. 初始学习率
   3. max_epochs：训练次数
   4. GPU配置，即调用GPU

2. 数据输入

    调用的是`Dataset`（用来重新调整数据的格式）和`DataLoade`（用迭代法iterative读入每个批次的数据）两个模块。

   层：
   - 卷积层：比如一个4x4的图，把其中3x3拿过来计算出一个数，用来代表这9个格子；如果格子不够，比如是个3x4的图，那么不够的部分会用0填充。具体的代码看[卷积层](2dCNN.py)

    里面有个东西叫做卷积核（通常3x3矩阵），不同的卷积核有不同的作用，实现一些功能，比如边缘检测，锐化，高斯模糊。
   - 池化层：把卷积层的东西再算一遍，减少数据量。[池化层](pool.py)

3. 有代表性的模型

   - LeNet
   
      前馈神经网络 (feed-forward network），接受一个输入再层层往后
     1. 定义包含一些可学习参数(或者叫权重）的神经网络；
     2. 在输入数据集上迭代；
     3. 通过网络处理输入
     4. 计算 loss (输出和正确答案的距离）
     5. 将梯度反向传播给网络的参数
     6. 更新网络的权重

   - AlexNet
       d

4. 模型初始化

要考虑好权重的初始值，好的值会让函数收敛得快。


5. 损失函数loss 也就是模型负反馈
   1. 二分类交叉熵损失函数(Cross Entropy): 计算交叉熵函数
   2. L1损失函数: 计算输出y和真实标签target之间的差值的绝对值。
   3. MSE损失函数: 计算输出y和真实标签target之差的平方。
   4. 平滑L1 (Smooth L1)损失函数: L1的平滑输出，其功能是减轻离群点带来的影响
   5. 目标泊松分布的负对数似然损失: 泊松分布的负对数似然损失函数
   6. KL散度:  计算KL散度，也就是计算相对熵。用于连续分布的距离度量，并且对离散采用的连续输出空间分布进行回归通常很有用。
   7. MarginRankingLoss: 计算两个向量之间的相似度，用于排序任务。该方法用于计算两组数据之间的差异。
   8. 多标签边界损失函数: 对于多标签分类问题计算损失函数。
   9. 二分类损失函数: 计算二分类的 logistic 损失。
   10. 多分类的折页损失
   11. 三元组损失
   12. HingEmbeddingLoss
   13. 余弦相似度
   14. CTC损失函数

6. PyTorch模型建立

此处要用到`torch.nn`里面的`nn.Module`模块。要定义一个个函数的主要流程是：1. 初始化；2. 定义数据流向。

主要方式`Sequential`，`ModuleList`和`ModuleDict`

在`Sequential`里面，有两个排列方法：
- Linear:
```
import torch.nn as nn
net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10), 
        )
```

- OrderedDict:
```
import collections
import torch.nn as nn
net2 = nn.Sequential(collections.OrderedDict([
          ('fc1', nn.Linear(784, 256)),
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(256, 10))
          ]))
```

## 启用tensor board
`tensorboard --logdir=/path/to/logs/ --port=xxxx`