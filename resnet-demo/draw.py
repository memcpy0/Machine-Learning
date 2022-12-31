"""
这份代码在train.py代码运行完毕（即网络训练完成后）再运行，会生成训练损失函数曲线和训练准确度曲线。
损失函数曲线应当是先快速下降然后逐渐收敛；训练准确度曲线应当是先快速上升然后逐步平稳。
"""
import json
import matplotlib.pyplot as plt

with open('training_statistic.json', 'r') as f:
    statistics = json.load(f)

loss, accuracy = statistics['loss'], statistics['accuracy']

plt.figure(1)
plt.plot(range(len(loss)), loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss curve of training')
plt.savefig('train_loss.png', dpi=300)
plt.show()

plt.figure(2)
plt.plot(range(len(accuracy)), accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy curve of training')
plt.savefig('train_accuracy.png', dpi=300)
plt.show()
