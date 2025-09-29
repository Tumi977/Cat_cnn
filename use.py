import os
import sys
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("当前设备:", device)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=128 * 8 * 8, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


model = SimpleCNN().to(device)

MODEL_PATH = r'cat_dog_cnn.pth'

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print(f"成功读取已保存的参数文件: {MODEL_PATH}，将以此为参数进行训练。")
else:
    print(f"未找到参数文件: {MODEL_PATH}。")
    sys.exit(1)

model.eval()
CLASS_NAMES = ['cat', 'dog']

inference_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"错误：未找到测试图像文件: {image_path}")
        return

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"错误：无法加载或处理图像 {image_path}: {e}")
        return

    input_tensor = inference_transform(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    if input_batch.dim() != 4:
        print(f"内部错误: 输入张量维度不正确 (预期 4, 实际 {input_batch.dim()})")
        return

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.softmax(output, dim=1)
    max_prob, predicted_index = torch.max(probabilities, 1)  # 返回最大概率和索引值

    predicted_class = CLASS_NAMES[predicted_index.item()]  # 是猫还是狗
    confidence = max_prob.item()

    print("\n================== 预测结果 ==================")
    print(f"图片: {os.path.basename(image_path)}")
    print(f"预测类别为: {predicted_class}")
    print(f"可信度: {confidence:.4f} ({confidence * 100:.2f}%)")
    print("==============================================")


TEST_IMAGE_PATH = r'F:\Cat_cnn\D2.jpg'

predict_image(TEST_IMAGE_PATH)
