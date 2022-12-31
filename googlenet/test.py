# -*- coding: utf-8 -*-
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from glob import glob
from googlenet_model import GoogLenet, InceptionAux

# 定义可以使用的设备
# 如果有NVIDA显卡,转到GPU训练，否则用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 图像数据转换 
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 获取存放测试图像的路径
test_paths = r"/kaggle/input/seashiptrainandtest/data/val/*/*.png"
# 通过库函数glob读取指定路径下所有符合匹配条件的文件（图片）
img_path_list = glob(test_paths, recursive=True)

class_indict = {"0": "sea", "1": "ship"}
class_indict_reverse = {v: k for k, v in class_indict.items()}
    
ground_truths = [int(class_indict_reverse[x.split('/')[-2]]) for x in img_path_list]

# 构建Googlenet模型
model = GoogLenet(num_classes=2, aux_logits=False)
model_weight_path = "/kaggle/working/googleNet2.pth"
model.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
model.fc1 = torch.nn.Linear(1024, 2)
model.aux1 = InceptionAux(512, 2)
model.aux2 = InceptionAux(528, 2)
model.to(device)
# 每次预测时将多少张图片打包成一个batch
batch_size = 32
# 统计整个测试过程中的TP/TN/FP/FN样本的总数
TPs, TNs, FPs, FNs = 0, 0, 0, 0

with torch.no_grad():  # PyTorch框架在验证网络性能时常用，无需像训练过程中记录网络的梯度数据
    for ids in range(0, round(len(img_path_list) / batch_size)):  # 根据批的划分情况，分批向网络传送数据
        # img_path_list中的元素只是图像的地址，下面将一批图像地址逐一读取为图像
        img_list = []
        # 由于是一次读取一批数据，所以可能存在批大小划分无法刚好划分完全部数据的情况，所以要放置下标越界的错误出现
        start = ids * batch_size
        end = -1 if (ids + 1) * batch_size >= len(img_path_list) else (ids + 1) * batch_size
        for img_path in img_path_list[start: end]:
            img = Image.open(img_path)
            img = data_transform(img)
            img_list.append(img)
        batch_ground_truths = ground_truths[start: end]  # 获取该批大小对应图片的类别序号
        # img_list内部即是一个批大小的图像数据，但是在输入网络之前我们需要将列表类型的数据转换为PyTorch支持的张量数据
        batch_img = torch.stack(img_list, dim=0)
        
        # 将数据输入网络，获得分类结果
        output = model(batch_img.to(device)).cpu()
        predict = torch.softmax(output, dim=1)
        # print(predict)
        
        probs, classes = torch.max(predict, dim=1)
        # 打印这一个批大小的数据的分类信息
        # for idx, (pro, cla) in enumerate(zip(probs, classes)):
        #    print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
        #                                                     class_indict[str(cla.numpy())],
        #                                                     pro.numpy()))
        batch_predicted_clses = classes.numpy().tolist()  # 对网络预测结果的变量类型做改变，变为列表

        # 计算该批下数据的TP/TN/FP/FN样本数量。如果符合条件的话，就向列表中添加一个1，然后计算整个列表的和值即为样本量
        TP = sum([1 for g, p in zip(batch_ground_truths, batch_predicted_clses) if g == p == 1])
        TN = sum([1 for g, p in zip(batch_ground_truths, batch_predicted_clses) if g == p == 0])
        FP = sum([1 for g, p in zip(batch_ground_truths, batch_predicted_clses) if g == 0 and p == 1])
        FN = sum([1 for g, p in zip(batch_ground_truths, batch_predicted_clses) if g == 1 and p == 0])

        # 一个批的预测结果加到总数上
        TPs += TP
        TNs += TN
        FPs += FP
        FNs += FN

# 根据定义，计算总数的各项指标
print("TPs: {} TNs: {} FPs: {} FNs: {}".format(TPs, TNs, FPs, FNs))
accuracy = (TNs + TPs) / len(img_path_list)
precision = TPs / (TPs + FPs)
recall = TPs / (TPs + FNs)
f1 = 2 * precision * recall / (precision + recall)
print(f'Overall performance:\n'
      f'Accuracy: {accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}')