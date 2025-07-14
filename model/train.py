import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime

# ✅ 설정
DATA_DIR = "data/processed"
MODEL_PATH = "model/emotion_model.pth"
LOG_DIR = "logs"

# ✅ 로그 파일 생성
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"emotion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
log_file = open(log_filename, "w")

def log(msg):
    print(msg)
    log_file.write(msg + "\n")

# ✅ 데이터 전처리
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ 데이터셋 불러오기
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

log(f"🔹 클래스 목록: {train_dataset.classes}")
log(f"🔹 학습 데이터: {len(train_dataset)}개, 테스트 데이터: {len(test_dataset)}개")

# ✅ 모델 정의
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

# ✅ 평가 함수
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ✅ 학습 시작
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 3  # 🔁 테스트 목적 에폭 축소
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        log(f"\n🟡 [Epoch {epoch+1}/{EPOCHS}] 시작")
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % 10 == 0:
                log(f"🔁 Step {step+1}/{len(train_loader)} – Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        log(f"✅ [Epoch {epoch+1}] 평균 Loss: {avg_loss:.4f}")

    # ✅ 모델 저장
    torch.save(model.state_dict(), MODEL_PATH)
    log(f"\n✅ 모델 저장 완료: {MODEL_PATH}")

    # ✅ 테스트 정확도 평가
    accuracy = evaluate(model, test_loader)
    log(f"📊 테스트 정확도: {accuracy:.2f}%")

    log_file.close()
