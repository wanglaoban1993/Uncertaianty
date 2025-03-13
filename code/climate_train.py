import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from torch.utils.tensorboard import SummaryWriter

# 1. 数据准备
# 读取 CSV 数据
data = pd.read_csv('../data/testset.csv')
print(data.head())
print(data.info())
features = data.iloc[:, 1:-1].values  # 假设第0列是时间，最后一列是温度
target = data.iloc[:, -1].values.reshape(-1, 1)  # 温度

# 数据归一化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
features_scaled = scaler_x.fit_transform(features)
target_scaled = scaler_y.fit_transform(target)

# 将数据转化为 LSTM 的输入格式
# def create_sequences(data_x, data_y, seq_length):
#     x, y = [], []
#     for i in range(len(data_x) - seq_length):
#         x.append(data_x[i:i + seq_length])
#         y.append(data_y[i + seq_length])
#     return np.array(x), np.array(y)

# seq_length = 10  # 序列长度
# x, y = create_sequences(features_scaled, target_scaled, seq_length)

def create_sequences_multistep(data_x, data_y, seq_length, forecast_steps):
    x, y = [], []
    for i in range(len(data_x) - seq_length - forecast_steps + 1):
        x.append(data_x[i:i + seq_length])
        y.append(data_y[i + seq_length:i + seq_length + forecast_steps])
    return np.array(x), np.array(y)

# 动态时间步长与多步预测
seq_length = 10  # 输入序列长度
forecast_steps = 3  # 预测未来的时间步长
x, y = create_sequences_multistep(features_scaled, target_scaled, seq_length, forecast_steps)


# 分割数据集
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 转换为 PyTorch 的 Tensor
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# 2. 模型构建
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         _, (hidden, _) = self.lstm(x)
#         out = self.fc(hidden[-1])
#         return out
class LSTMModelMultiStep(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, forecast_steps):
        super(LSTMModelMultiStep, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, forecast_steps)  # 输出预测多个时间步

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

# # 增加模型超参数调整
# def build_model(input_size, hidden_size=64, num_layers=2, dropout=0.2):
#     return LSTMModel(input_size, hidden_size, num_layers, dropout)
# 动态构建模型
def build_model_multistep(input_size, hidden_size=64, num_layers=2, dropout=0.2, forecast_steps=1):
    return LSTMModelMultiStep(input_size, hidden_size, num_layers, dropout, forecast_steps)

# 3. 模型训练和保存
# def train_model_with_saving(model, x_train, y_train, x_val, y_val, num_epochs, batch_size, save_path):
#     train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
#     val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     # writer = SummaryWriter(log_dir)  # 初始化 TensorBoard
#     best_val_loss = float('inf')

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         for x_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(x_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         # 验证阶段
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for x_batch, y_batch in val_loader:
#                 outputs = model(x_batch)
#                 loss = criterion(outputs, y_batch)
#                 val_loss += loss.item()

#         avg_train_loss = train_loss / len(train_loader)
#         avg_val_loss = val_loss / len(val_loader)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

#         # 将损失写入 TensorBoard
#         writer.add_scalar('Loss/Train', avg_train_loss, epoch+1)
#         writer.add_scalar('Loss/Validation', avg_val_loss, epoch+1)

#         # 保存最佳模型
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), save_path)
#             print(f"Model saved at epoch {epoch+1} with val loss: {best_val_loss:.4f}")

# # 模型训练
# save_path = "../result/best_lstm_model.pth"
# model = build_model(input_size=features.shape[1], hidden_size=64, num_layers=2, dropout=0.2)
# train_model_with_saving(model, x_train, y_train, x_val, y_val, num_epochs=50, batch_size=32, save_path=save_path)

def train_model_multistep(model, x_train, y_train, x_val, y_val, num_epochs, batch_size, save_path):
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with val loss: {best_val_loss:.4f}")

# 训练多步预测模型
save_path = "best_multistep_lstm_model.pth"
model = build_model_multistep(input_size=features.shape[1], hidden_size=64, num_layers=2, dropout=0.2, forecast_steps=forecast_steps)
train_model_multistep(model, x_train, y_train, x_val, y_val, num_epochs=50, batch_size=32, save_path=save_path)

# 4. 测试阶段：多步预测与对比绘图
def test_and_plot_multistep(saved_model_path, x_test, y_test, scaler_y, forecast_steps):
    model = build_model_multistep(input_size=features.shape[1], hidden_size=64, num_layers=2, dropout=0.2, forecast_steps=forecast_steps)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    with torch.no_grad():
        test_predictions = model(x_test).numpy()
        test_predictions = scaler_y.inverse_transform(test_predictions)  # 反归一化
        y_test = scaler_y.inverse_transform(y_test.numpy())

        # 绘制预测值与真实值对比
        plt.figure(figsize=(12, 6))
        for i in range(forecast_steps):
            plt.plot(y_test[:, i], label=f"Actual (Step {i+1})", linestyle="dashed")
            plt.plot(test_predictions[:, i], label=f"Predicted (Step {i+1})")
        plt.legend()
        plt.title(f"Multi-Step Prediction vs Actual Values ({forecast_steps} Steps)")
        plt.xlabel("Time Step")
        plt.ylabel("Temperature")
        plt.show()

    test_mse = np.mean((test_predictions - y_test) ** 2)
    print(f"Test MSE: {test_mse:.4f}")
    return test_predictions, y_test

# 测试多步预测
test_predictions, y_test_actual = test_and_plot_multistep(save_path, x_test, y_test, scaler_y, forecast_steps)