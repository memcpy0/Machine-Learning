# -*- coding: utf-8 -*-
import os
import random
import shutil
from glob import glob
from pathlib import Path

# 这份代码用于将原始数据集进行划分，变为PyTorch接口可以直接使用的训练集和测试集
def processor(dataset_root: str, train_ratio: float = 0.8):
    """
    将原始数据集划分为训练集和测试集。在使用该函数前，需要保证所有的船舶分类图像数据已经存储在data目录下，
    并且按照类别分好类（data/sea/文件夹内有3000张非船舶图片，data/ship/文件夹内有1000张船舶图片）。

    划分数据集的结果是：data文件夹下新建两个文件夹，分别是train和val文件夹。这两个文件夹中都存放有sea和ship文件夹，
    理论上data/train/sea/中有2400张图片，data/train/ship/中有800张图片；
    同理，data/val/sea/中有600张图片，data/val/ship/中共有200张图片。

    :param dataset_root: data文件夹的路径.
    :param train_ratio: 训练集的占比。按照8:2划分数据集的话，这个参数应当设置为0.8.
    :return: None
    """
    # 确保所有输入的参数没有问题
    assert os.path.exists(dataset_root) and os.path.isdir(dataset_root), \
        'Invalid dataset root directory!'
    assert 0.5 < train_ratio < 1, 'Invalid trainset ratio!'

    # 获取所有的正负样本。正样本即船舶图像，负样本即非船舶图像
    neg_samples = glob(os.path.join(dataset_root, 'sea/*.png'), recursive=False)  # 获取所有负样本的相对路径
    random.shuffle(neg_samples)  # 打乱读取的负样本的顺序，增加随机性
    pos_samples = glob(os.path.join(dataset_root, 'ship/*.png'), recursive=False)  # 获取所有正样本的相对路径
    random.shuffle(pos_samples)  # 打乱读取的正样本的顺序，增加随机性
    num_neg_samples = len(neg_samples)  # 获取正负样本的数量
    num_pos_samples = len(pos_samples)
    # print(num_neg_samples, num_pos_samples) # 3000 1000

    # 根据训练集的比例，计算从负样本中抽取的、作为训练集的负样本数量
    # 根据这个数量，随机从负样本中采样作为训练集，然后剩余样本作为负样本的测试集
    num_neg_training_samples = round(num_neg_samples * train_ratio)  # 计算负样本用于训练的样本数量
    neg_training_samples = random.sample(neg_samples, num_neg_training_samples)  # 对负样本训练集进行随机采样
    neg_testing_samples = [x for x in neg_samples if x not in neg_training_samples]  # 剩下的负样本作为测试集

    # 同理对正样本做训练集和测试集的采样
    num_pos_training_samples = round(num_pos_samples * train_ratio)
    pos_training_samples = random.sample(pos_samples, num_pos_training_samples)
    pos_testing_samples = [x for x in pos_samples if x not in pos_training_samples]

    # 完成正负样本的训练集测试集采样后，我们需要将原始没有划分好的数据组织为一个划分好的结构
    # 首先组织训练集
    train_dir = os.path.join(dataset_root, 'train/')  # 在data文件夹下新建一个train文件夹
    if os.path.exists(train_dir):  # 保证这个文件夹是空的
        shutil.rmtree(train_dir)
    os.makedirs(os.path.join(train_dir, 'sea/'))  # 在train文件夹下新建两个代表不同类别的文件夹，用于存储两类图像数据
    os.makedirs(os.path.join(train_dir, 'ship/'))
    for neg_train_sample in neg_training_samples:  # 将非船舶类别的训练图像数据复制到训练集中的sea文件夹中
        shutil.copyfile(
            src=neg_train_sample,
            dst=os.path.join(dataset_root, f'train/sea/{Path(neg_train_sample).name}')
        )
    for pos_train_sample in pos_training_samples:  # 将船舶类别的训练图像数据复制到训练集中的ship文件夹中
         shutil.copyfile(
            src=pos_train_sample,
            dst=os.path.join(dataset_root, f'train/ship/{Path(pos_train_sample).name}')
        )

    # 完成对训练集的组织后，组织测试集
    test_dir = os.path.join(dataset_root, 'val/')  # 同样在data文件夹下新建一个val文件夹，用于存储测试集数据
    if os.path.exists(test_dir):  # 确保文件夹为空
        shutil.rmtree(test_dir)
    os.makedirs(os.path.join(test_dir, 'sea/'))  # 同样代表两个类的文件夹
    os.makedirs(os.path.join(test_dir, 'ship/'))
    for neg_test_sample in neg_testing_samples:
        shutil.copyfile(
            src=neg_test_sample,
            dst = os.path.join(dataset_root, f'val/sea/{Path(neg_test_sample).name}')
        )
    for pos_test_sample in pos_testing_samples:
        shutil.copyfile(
            src=pos_test_sample,
            dst=os.path.join(dataset_root, f'val/ship/{Path(pos_test_sample).name}')
        )
    # 数据集的划分任务即完成

if __name__ == "__main__":
    processor(dataset_root='data/')
