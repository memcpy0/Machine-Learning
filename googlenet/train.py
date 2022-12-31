# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from googlenet_model import GoogLenet, InceptionAux

# 设置gpu训练模型
# 如果有NVIDA显卡，转到GPU训练，否则用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("using {} device.".format(device))
# 数据预处理
trans_dic = {
    # Compose()：将多个transforms的操作整合在一起
    # 训练
    "train": transforms.Compose([
        # RandomResizedCrop(224)：将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为给定大小
        transforms.RandomResizedCrop(224),
        # RandomVerticalFlip()：以0.5的概率竖直翻转给定的PIL图像
        transforms.RandomHorizontalFlip(),
        # ToTensor()：数据转化为Tensor格式
        transforms.ToTensor(),
        # Normalize()：将图像的像素值归一化到[-1,1]之间，使模型更容易收敛
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    # 验证
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# 读取数据
# 加载训练数据集
# ImageFolder：假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名，其构造函数如下：
# ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
# root：在指定路径下寻找图片，transform：对PILImage进行的转换操作，输入是使用loader读取的图片
train_set = datasets.ImageFolder(
    root=r"/kaggle/input/seashiptrainandtest/data/train",
    transform=trans_dic["train"])
# 训练集长度
train_num = len(train_set)
# 一次训练载入32张图像
batch_size = 32
# 确定进程数
# min() 返回给定参数的最小值，参数可以为序列
# cpu_count() 返回一个整数值，表示系统中的CPU数量，如果不确定CPU的数量，则不返回任何内容
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))
# DataLoader 将读取的数据按照batch size大小封装给训练集
# dataset (Dataset) 输入的数据集
# batch_size (int, optional): 每个batch加载多少个样本，默认: 1
# shuffle (bool, optional): 设置为True时会在每个epoch重新打乱数据，默认: False
# num_workers(int, optional): 决定了有几个进程来处理，默认为0意味着所有的数据都会被load进主进程
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)

# 加载测试数据集
validate_set = datasets.ImageFolder(
    root=r"/kaggle/input/seashiptrainandtest/data/val",
    transform=trans_dic["val"])
# 测试集长度
validate_num = len(validate_set)
validate_loader = DataLoader(validate_set, batch_size=batch_size, shuffle=True, num_workers=nw)
print("using {} images for training, {} images for validation.".format(train_num, validate_num))

# 模型实例化，将模型转到device, 二分类
net = GoogLenet(num_classes=2, aux_logits=True, init_weights=True)
# model_weight_path = "/kaggle/working/googlenet-pre.pth"
# net.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
# net.fc1 = torch.nn.Linear(1024, 2)
# net.aux1 = InceptionAux(512, 2)
# net.aux2 = InceptionAux(528, 2)
net.to(device)
# 定义损失函数（交叉熵损失）
loss_function = nn.CrossEntropyLoss()
# 定义adam优化器
# params(iterable) 要训练的参数，一般传入的是model.parameters()
# lr(float): learning_rate学习率，也就是步长，默认：1e-3
# params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(net.parameters(), lr=0.0003)

# 用于判断最佳模型
best_acc = 0.0
# 迭代次数（训练次数）
epoches = 15
# 最佳模型保存地址
save_path = "/kaggle/working/googleNet1.pth"
train_step = len(train_loader)
loss_r, acc_r = [], []  # 记录训练时出现的损失以及分类准确度
# 开始训练
for epoch in range(epoches):
    net.train()
    running_loss = 0.0
    # tqdm 进度条显示
    train_bar = tqdm(train_loader, file=sys.stdout)
    # train_bar 传入数据（数据包括训练数据和标签）
    # enumerate() 将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for循环当中
    # enumerate返回值: 一个是序号，一个是数据（包含训练数据和标签）
    # x: 训练数据（inputs）(tensor类型的）, y: 标签（labels）(tensor类型）
    for step, data in enumerate(train_bar):
        # 前向传播
        train_inputs, train_labels = data
        # 计算训练值
        train_outputs, aux_logits2, aux_logits1 = net(train_inputs.to(device))
        # GoogLeNet的网络输出loss有三个部分，分别是主干输出loss、两个辅助分类器输出loss（权重0.3）
        loss0 = loss_function(train_outputs, train_labels.to(device))
        loss1 = loss_function(aux_logits2, train_labels.to(device))
        loss2 = loss_function(aux_logits1, train_labels.to(device))
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3

        # 反向传播
        # 清空过往梯度
        optimizer.zero_grad()
        # 反向传播，计算当前梯度
        loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        # item()：得到元素张量的元素值
        running_loss += loss.item()

        # 进度条的前缀
        # .3f：表示浮点数的精度为3（小数位保留3位）
        train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1, epoches, loss)

    loss_r.append(running_loss)  # 该损失数值会被保存起来
    # 测试
    # eval()：如果模型中Batch Normalization和Dropout，则不启用，以防改变权值
    net.eval()  # validate
    acc = 0.0
    # 清空历史梯度，与训练最大的区别是测试过程中取消了反向传播
    with torch.no_grad():
        test_bar = tqdm(validate_loader, file=sys.stdout)
        for data in test_bar:
            test_inputs, test_labels = data
            test_outputs = net(test_inputs.to(device))
            # torch.max(input, dim)函数
            # input是具体的tensor，dim是max函数索引的维度，0是每列的最大值，1是每行的最大值输出
            # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
            predict_y = torch.max(test_outputs, dim=1)[1]
            # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False
            # .sum()对输入的tensor数据的某一维度求和
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

        val_accurate = acc / validate_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print("[epoch {}] train_loss:{:.3f}, val_accuracy:{:.3f} ".format(epoch + 1, running_loss / train_step, val_accurate))

    acc_r.append(val_accurate)  # 保存该准确度信息

with open('/kaggle/working/training_statistic_googlenet_v1.json', 'w+') as f:  # 保存训练过程的损失和准确度数据为一个json文件
    json.dump(dict(loss=loss_r, accuracy=acc_r), f, indent=4)

print("Finish training!")