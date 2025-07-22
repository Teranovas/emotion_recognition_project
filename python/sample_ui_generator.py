# sample_ui_generator.py
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os

# ✅ 출력 경로
os.makedirs("assets", exist_ok=True)
output_path = "assets/sample_ui.png"

# ✅ 빈 화면 생성 (640x480)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
frame[:] = (30, 30, 30)  # 약간 어두운 배경

# ✅ 감정 예시 (박스 및 라벨)
face_coords = [(150, 100, 250, 250), (350, 120, 450, 270)]
labels = ["기쁨", "슬픔"]
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = ImageFont.truetype(font_path, 28)

# ✅ 감정 카운트 예시
emotion_counts = {"기쁨": 4, "슬픔": 2, "화남": 1}
count_text = "   ".join([f"{k} {v}" for k, v in emotion_counts.items()])

# ✅ PIL로 텍스트 추가
image_pil = Image.fromarray(frame)
draw = ImageDraw.Draw(image_pil)
draw.text((10, 10), count_text, font=font, fill=(255, 255, 255))

# ✅ 얼굴 박스 + 감정 라벨
for (x1, y1, x2, y2), label in zip(face_coords, labels):
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    draw.text((x1, y1 - 30), label, font=font, fill=(0, 255, 0))

# ✅ 저장
image_bgr = np.array(image_pil)
cv2.imwrite(output_path, image_bgr)
print(f"✅ 샘플 UI 이미지 저장 완료: {output_path}")
