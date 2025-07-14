import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ✅ 감정 라벨
CLASS_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ✅ 모델 정의 (학습 구조와 동일)
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ✅ 이미지 배열(Numpy or 경로) 처리 가능하도록 개선
def predict_emotion(image_input, model_path="model/emotion_model.pth") -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCNN(num_classes=len(CLASS_LABELS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ✅ 입력이 ndarray인 경우 (웹캠, 실시간 이미지)
    if isinstance(image_input, np.ndarray):
        img_pil = Image.fromarray(image_input)
    else:
        # 파일 경로일 경우
        img_pil = Image.open(image_input).convert("RGB")

    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted = torch.argmax(outputs, dim=1).item()

    return CLASS_LABELS[predicted]
