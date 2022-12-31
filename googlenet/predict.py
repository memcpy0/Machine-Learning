# -*- coding: utf-8 -*-
import os
import json
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from googlenet_model import GoogLenet
# 定义可以使用的设备
# 如果有NVIDA显卡,转到GPU训练，否则用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 图像数据转换
# data_transform = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(224),
#      transforms.ToTensor(),
#      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 获取测试图像的路径
img_path = r"/kaggle/input/seashiptrainandtest/data/val/ship/ship__20170609_180756_103a__-122.33804952436104_37.737959994177224.png"
img = Image.open(img_path)
# imshow()对图像进行处理并显示其格式，show()则是将imshow()处理后的函数显示出来
plt.imshow(img)
# [C, H, W]，转换图像格式
img = data_transform(img) # [N, C, H, W]
# [N, C, H, W]，增加一个维度N
img = torch.unsqueeze(img, dim=0) # expand batch dimension

# class_indict = {"0": "sea", "1": "ship"}
# class_indict_reverse = {v: k for k, v in class_indict.items()}
class_indict = {"sea": '0', "ship": '1'}
class_indict_reverse = {v: k for k, v in class_indict.items()}

# 模型实例化，将模型转到device，结果类型有5种
# 实例化模型时不需要辅助分类器
model = GoogLenet(num_classes=2, aux_logits=False).to(device)
# 载入模型权重
weights_path = "/kaggle/working/googleNet2.pth"
# 在加载训练好的模型参数时，由于其中是包含有辅助分类器的，需要设置strict=False舍弃不需要的参数
missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
# 进入验证阶段
model.eval()
with torch.no_grad():
    # 预测类别
    # squeeze() 维度压缩，返回一个tensor（张量），其中input中大小为1的所有维都已删除
    output = torch.squeeze(model(img.to(device))).cpu()
    # softmax 归一化指数函数，将预测结果输入进行非负性和归一化处理，最后将某一维度值处理为0-1之内的分类概率
    predict = torch.softmax(output, dim=0)
    # argmax(input)返回指定维度最大值的序号
    # .numpy()把tensor转换成numpy的格式
    predict_class = torch.argmax(predict).numpy()

# 输出的预测值与真实值
print_res = "class: {}   prob: {:.3}".format(class_indict_reverse[str(predict_class)], predict[predict_class].numpy())
# 图片标题
plt.title(print_res)
for i in range(len(predict)):
    print("class: {:10}   prob: {:.3}".format(class_indict_reverse[str(i)], predict[i].numpy()))
plt.show()