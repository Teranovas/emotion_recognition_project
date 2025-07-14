import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 감정 라벨 리스트
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 모델 구조 정의 (예시 CNN 구조 - 학습 시 사용한 것과 동일해야 함)
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, len(EMOTIONS))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 추론 함수
def predict_emotion(face_img: np.ndarray, model_path: str = "model/emotion_model.pth") -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 전처리
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img_pil = Image.fromarray(face_img)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)  # (1, 1, 48, 48)

    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()

    return EMOTIONS[predicted]

if __name__ == "__main__":
    import cv2

    # 테스트용 흑백 이미지 경로 (예: 48x48 얼굴 이미지)
    test_img = cv2.imread("data/raw/sample_face.jpg", cv2.IMREAD_GRAYSCALE)

    if test_img is None:
        print("❌ 이미지 파일을 불러올 수 없습니다. data/raw/sample_face.jpg 를 확인해주세요.")
    else:
        result = predict_emotion(test_img)
        print("🎯 감정 추론 결과:", result)
