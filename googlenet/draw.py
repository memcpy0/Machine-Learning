# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt

loss_path1=r"/kaggle/working/training_statistic_googlenet_v1.json"
loss_path2=r"/kaggle/working/training_statistic_googlenet_v2.json"
loss_path3=r"/kaggle/working/training_statistic_googlenet_v3.json"
with open(loss_path1, 'r') as f:
    statistics1 = json.load(f)
with open(loss_path2, 'r') as f:
    statistics2 = json.load(f)
with open(loss_path3, 'r') as f:
    statistics3 = json.load(f)

loss1, accuracy1 = statistics1['loss'], statistics1['accuracy']
loss2, accuracy2 = statistics2['loss'], statistics2['accuracy']
loss3, accuracy3 = statistics3['loss'], statistics3['accuracy']

plt.figure(1)
plt.plot(range(len(loss1)), loss1, color="#F52A2A", linestyle="-", label="GoogLeNet_v1: adam+0.0003")
plt.plot(range(len(loss2)), loss2, color="#FFC000", linestyle="-", label="GoogLeNet_v2: adam+0.003")
plt.plot(range(len(loss3)), loss3, color="#FFC0F0", linestyle="-", label="GoogLeNet_v3: adam+0.001")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss curve of training')
plt.savefig('train_loss_comparsion.png', dpi=600)
plt.show()
#
plt.figure(1)
plt.plot(range(len(accuracy1)), accuracy1, color="#F52A2A", linestyle="-", label="GoogLeNet_v1: adam+0.0003")
plt.plot(range(len(accuracy2)), accuracy2, color="#FFC000", linestyle="-", label="GoogLeNet_v2: adam+0.003")
plt.plot(range(len(accuracy3)), accuracy3, color="#FFC0F0", linestyle="-", label="GoogLeNet_v3: adam+0.001")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy curve of training')
plt.savefig('train_accuracy.png', dpi=600)
plt.show()