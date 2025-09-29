import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

# ==================== 0. 设备选择 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("当前设备:", device)

# ==================== 1. 数据准备 ====================
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 可以改成224x224，但显存需求大
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 使用 ImageFolder 自动读取文件夹
full_dataset = datasets.ImageFolder(root='train', transform=transform)

# 划分训练集/验证集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")


# ==================== 2. 定义 CNN 模型 ====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # in_channels代表输入数据的深度,out_channels代表输出的维度，
            # 输出层为32维代表了32个卷积层，每一个对输入图像处理后是一个结果（特征图）。padding即为在输入周围填充 1 层0，但是没有在深度（channel）上进行填充
            # 卷积核就像全连接神经网络中的一个神经元，只不过它能实现对某一特定位置（很小很精确的位置）的特征的高度敏感，而非是像神经元那样需要从全图中识别某些特征
            # 这个体积内包含了 3×3×3=27 个可学习的权重（Weight），这些权重就是模型在训练中学习的参数。
            # 此外，这个卷积核还有一个额外的偏置项（Bias），通常是一个单一的数值。
            # 这个最终的单一数值就是输出特征图上的一个像素点。这个值越大，说明该局部区域越强烈地包含了该卷积核想要提取的特征。
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 池化窗口大小（kernel_size）是 2×2，并且默认步长（stride）也为 2，一个 2×2 的窗口在输入特征图上滑动。
            # 窗口每次覆盖 4 个数值。池化操作会从这 4 个数值中只选择最大的那个数值作为输出。
            # ↓ ： 传给下一层的数据就是 32（channels） * 32 * 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ↓ ： 传给下一层的数据就是 64（channels） * 16 * 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # 结束 ， 大小为 128 * 8 * 8
        )
        # ↓：由特征提取转向决策
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),  # 输入尺寸取决于图片大小
            # 这是一个标准的矩阵乘法和加偏置操作。它将 8192 个原始特征进行加权组合和压缩，提炼出 256 个更高级、更紧凑的中间特征。
            # 跟全神经连接网络一个原理，用256个神经元对提取出来的可能是某个特定特征的数据进行敏感度加权
            nn.ReLU(),
            # 能够对 8192 个原始特征进行非线性组合和转换，生成更高级、更复杂的 256 个非线性特征。
            nn.Linear(256, 2)  # 输出2类

        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


model = SimpleCNN().to(device)

# ==================== 3. 损失函数与优化器 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
MODEL_PATH = 'cat_dog_cnn.pth'

# --- 新增功能：读取保存的参数 ---
if os.path.exists(MODEL_PATH):
    # 仅加载模型参数（state_dict）
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"成功读取已保存的参数文件: {MODEL_PATH}，将从上次进度继续训练。")
else:
    print(f"未找到参数文件: {MODEL_PATH}，将从头开始训练。")
# ---------------------------------

# ==================== 4. 训练循环 ====================
num_epochs = 50
max_val_acc = 0.0  # 追踪历史最高验证准确率
best_epoch_info = {} # 记录最佳周期的所有指标

for epoch in range(num_epochs):
    # ----训练----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    # ----验证----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")
    
    # --- 最佳模型保存逻辑 ---
    if val_acc > max_val_acc:
        # 更新最高准确率
        max_val_acc = val_acc
        
        # 记录当前周期所有指标
        best_epoch_info = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        # 只有在创下新高时才保存模型
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"-> 【新纪录！】Val Acc: {max_val_acc:.4f}. 已保存最佳模型到 '{MODEL_PATH}'")

# --- 训练结束后，打印最佳结果 ---
if best_epoch_info:
    print("\n==============================================")
    print("              训练结束，最佳结果：")
    print("==============================================")
    print(f"最佳周期 (Epoch): {best_epoch_info['epoch']}")
    print(f"最高验证准确率 (Val Acc): {best_epoch_info['val_acc']:.4f}")
    print(f"对应验证损失 (Val Loss): {best_epoch_info['val_loss']:.4f}")
    print(f"对应训练准确率 (Train Acc): {best_epoch_info['train_acc']:.4f}")
    print("==============================================")
else:
    # 适用于 max_val_acc 没有更新的情况（例如第一个 epoch 就因某种原因失败）
    print("\n训练结束，但未找到有效的最佳模型记录。")