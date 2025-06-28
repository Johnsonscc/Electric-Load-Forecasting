import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_model import TimeSeriesDataset, LSTMModel
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 读取测试集的数据
test_df = pd.read_csv('./dataset/huihedata_month.csv')

# 将数据集转换为PyTorch的Tensor
test_data = torch.tensor(test_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)

# 创建测试集的数据集对象
seq_length = 6
test_dataset = TimeSeriesDataset(test_data, seq_length)

# 创建数据加载器
batch_size = 2
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型参数
input_size = 1
hidden_size = 256
num_layers = 8
output_size = 1

# 创建模型实例
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载已训练好的模型参数
model.load_state_dict(torch.load('best_model.pt'))
model.to(device)
model.eval()  # 确保模型处于评估模式

# 在测试集上进行预测
predictions = []
true_values = test_df['Elia Grid Load [MW]'].values[seq_length:]

with torch.no_grad():
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.append(outputs.detach().cpu().numpy())

# 将预测结果转换为一维数组
predictions = np.concatenate(predictions).flatten()

# 计算误差指标
mse = mean_squared_error(true_values, predictions)
mae = mean_absolute_error(true_values, predictions)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# 绘制预测值和真实值的曲线图
plt.figure(figsize=(20, 8))  # 设置图形大小为50x8英寸
plt.plot(test_df.index, test_df['Elia Grid Load [MW]'], label='True Values')
plt.plot(test_df.index[seq_length:], predictions, label='Predicted Values')
plt.xlabel('Time')
plt.ylabel('Load [MW]')
plt.title('Predicted vs True Values')
plt.legend()
plt.show()
