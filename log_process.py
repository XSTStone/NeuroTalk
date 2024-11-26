import re
import matplotlib.pyplot as plt

# 读取log文件
log_file = 'D:\\NeuroTalk-main\\val_log.log'

# 初始化列表，用来存储训练和验证数据
epochs_train = []
g_rmse_train = []
epochs_val = []
g_rmse_val = []

# 读取log文件并提取数据
with open(log_file, 'r') as file:
    for line in file:
        # 匹配训练数据（例如：Epoch 1 Batch [73/74] g-RMSE: 0.3868）
        train_match = re.search(r'Epoch (\d+) Batch \[\d+/(\d+)\] g-RMSE: (\d+\.\d+)', line)
        if train_match:
            epoch = int(train_match.group(1))
            g_rmse = float(train_match.group(3))
            batch_total = int(train_match.group(2))
            if batch_total == 74:  # 训练数据的 Batch 是 73/74
                epochs_train.append(epoch)
                g_rmse_train.append(g_rmse)

        # 匹配验证数据（例如：Epoch 1 Batch [9/10] g-RMSE: 0.3292）
        val_match = re.search(r'Epoch (\d+) Batch \[\d+/(\d+)\] g-RMSE: (\d+\.\d+)', line)
        if val_match:
            epoch = int(val_match.group(1))
            g_rmse = float(val_match.group(3))
            batch_total = int(val_match.group(2))
            if batch_total == 10:  # 验证数据的 Batch 是 9/10
                epochs_val.append(epoch)
                g_rmse_val.append(g_rmse)

# 绘制训练数据的折线图
plt.figure(figsize=(10, 6))

# 绘制训练数据
plt.plot(epochs_train, g_rmse_train, label='Training g-RMSE', color='blue', marker='o')

# 绘制验证数据
plt.plot(epochs_val, g_rmse_val, label='Validation g-RMSE', color='red', marker='x')

# 添加标题和标签
plt.xlabel('Epoch')
plt.ylabel('g-RMSE')
plt.title('Training and Validation g-RMSE Over Epochs')

# 添加图例
plt.legend()

# 显示网格
plt.grid(True)

# 保存图像到文件
plt.tight_layout()
plt.savefig('val_g_rmse_epochs.png')  # 保存为 PNG 文件，您可以修改文件名

# 显示图形
plt.tight_layout()
plt.show()
