# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

#  基础卷积层Conv2d+ReLu
class BasicConv2d(nn.Module):
    # init: c进行初始化，申明模型中各层的定义
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        # ReLU(inplace=True): 将tensor直接修改，不找变量做中间的传递，节省运算内存，不用多存储额外的变量
        self.relu = nn.ReLU(inplace=True)

    # 前向传播过程
    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        return x

# Inception结构
class Inception(nn.Module):
    def __init__(self, in_channels, ch1_1, ch3_3red, ch3_3, ch5_5red, ch5_5, pool_pro):
        super(Inception, self).__init__() # 1_1 => 1x1
        # 分支1，单1x1卷积层
        # input:[in_channels,height,weight],output:[ch1_1,height,weight]
        self.branch1 = BasicConv2d(in_channels=in_channels, out_channels=ch1_1, kernel_size=1, stride=1)
        # input:[in_channels,height,weight],output:[ch3_3,height,weight]
        # 分支2，1x1卷积层后接3x3卷积层
        self.branch2 = nn.Sequential(
            # input:[in_channels,height,weight],output:[ch3_3red,height,weight]
            BasicConv2d(in_channels=in_channels, out_channels=ch3_3red, kernel_size=1, stride=1),
            # input:[ch3_3red,height,weight],output:[ch3_3,height,weight]
            # 保证输出大小等于输入大小
            BasicConv2d(in_channels=ch3_3red, out_channels=ch3_3, kernel_size=3, stride=1, padding=1), # 保证输出大小等于输入大小
        )
        # 分支3，1x1卷积层后接5x5卷积层
        self.branch3 = nn.Sequential(
            # input:[in_channels,height,weight],output:[ch5_5red,height,weight]
            BasicConv2d(in_channels=in_channels, out_channels=ch5_5red, kernel_size=1, stride=1),
            # 在官方实现中是3x3kernel并不是5x5，具体参考issue——https://github.com/pytorch/vision/issues/906.
            # input:[ch5_5red,height,weight],output:[ch5_5,height,weight]
            # 保证输出大小等于输入大小
            BasicConv2d(in_channels=ch5_5red, out_channels=ch5_5, kernel_size=5, stride=1, padding=2), # 保证输出大小等于输入大小
        )
        # 分支4，3x3最大池化层后接1x1卷积层
        self.branch4 = nn.Sequential(
            # input:[in_channels,height,weight],output:[in_channels,height,weight]
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # input:[in_channels,height,weight],output:[pool_pro,height,weight]
            BasicConv2d(in_channels=in_channels, out_channels=pool_pro, kernel_size=1, stride=1),
        )
    # forward: 定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        output1 = self.branch1(x)
        output2 = self.branch2(x)
        output3 = self.branch3(x)
        output4 = self.branch4(x)
        # 在通道维上连结输出
        # cat()在给定维度上对输入的张量序列进行连接操作
        return torch.cat([output1, output2, output3, output4], dim=1)

# 辅助分类器: 4e和4a输出
class InceptionAux(nn.Module):
    def __init__(self, in_channels, class_num=1000):
        super(InceptionAux, self).__init__()
        # 4a:input:[512,14,14];output:[512,4,4]
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        # 4a:input:[512,4,4];output:[128,4,4]
        self.conv2d = BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        # 上一层output[batch, 128, 4, 4]，128X4X4=2048
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=class_num)

    # 前向传播过程
    def forward(self, x):
        # 输入：分类器1：Nx512x14x14，分类器2：Nx528x14x14
        x = self.averagePool(x)
        # 输入：分类器1：Nx512x14x14，分类器2：Nx528x14x14
        x = self.conv2d(x)
        # 输入：N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # 设置.train()时为训练模式，self.training=True
        x = F.dropout(x, p=0.5, training=self.training)
        # 输入：N x 2048
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=0.5, training=self.training)
        # 输入：N x 1024
        x = self.fc2(x)
        # 返回值：N*num_classes
        return x

# 定义GoogLeNet网络模型
class GoogLenet(nn.Module):
    # init: 进行初始化，申明模型中各层的定义
    # num_classes: 需要分类的类别个数
    # aux_logits: 训练过程是否使用辅助分类器
    # init_weights: 是否对网络进行权重初始化
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLenet, self).__init__()
        # 是否选择辅助分类器
        self.aux_logits = aux_logits
        # 构建网络
        # input:[3,224,224],output:[64,112,112],padding自动忽略小数
        self.conv1 = BasicConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        # input:[64,112,112],output:[64,56,56](55.5->56)
        # ceil_mode=true时，将不够池化的数据自动补足NAN至kernel_size大小
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # input:[64,56,56],output:[64,56,56]
        self.conv2 = BasicConv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        # input:[192,56,56],output:[192,56,56]
        self.conv3 = BasicConv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)

        # input:[192,56,56],output:[192,28,28](27.5->28)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # input:[192,28,28],output:[256,28,28]
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # input:[256,28,28],output:[480,28,28]
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        # input:[480,28,28],output:[480,14,14]
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # input:[480,14,14],output:[512,14,14]
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        # input:[512,14,14],output:[512,14,14]
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        # input:[512,14,14],output:[512,14,14]
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # input:[512,14,14],output:[528,14,14]
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        # input:[528,14,14],output:[832,14,14]
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        # input:[832,14,14],output:[832,7,7]
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # input:[832,7,7],output:[832,7,7]
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        # input:[832,7,7],output:[1024,7,7]
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 如果为真，则使用辅助分类器
        if self.aux_logits:
            self.aux1 = InceptionAux(512, class_num=num_classes) # 4a输出
            self.aux2 = InceptionAux(528, class_num=num_classes) # 4d输出

        # AdaptiveAvgPool2d：自适应平均池化，指定输出（H，W）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(in_features=1024, out_features=num_classes)
        # 如果为真，则对网络参数进行初始化
        if init_weights:
            self._initialize_weights()

    # forward() 定义前向传播过程,描述各层之间的连接关系
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        # 若存在辅助分类器
        if self.training and self.aux_logits: # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        # 若存在辅助分类器
        if self.training and self.aux_logits: # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = F.dropout(x, p=0.4)
        x = self.fc1(x)
        # N x 1000 (num_classes)
        if self.aux_logits and self.training:
            return x, aux2, aux1
        return x

    # 网络结构参数初始化
    def _initialize_weights(self):
        # 遍历网络中的每一层
        for m in self.modules():
            # isinstance(object, type)，如果指定的对象拥有指定的类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # Kaiming正态分布方式的权重初始化
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # 如果偏置不是0，将偏置置成0，对偏置进行初始化
                if m.bias is not None:
                    # torch.nn.init.constant_(tensor, val)，初始化整个矩阵为常数val
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # init.normal_(tensor, mean=0.0, std=1.0)，使用从正态分布中提取的值填充输入张量
                # 参数：tensor：一个n维Tensor，mean：正态分布的平均值，std：正态分布的标准差
                nn.init.uniform_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)