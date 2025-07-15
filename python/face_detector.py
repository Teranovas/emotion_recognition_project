import traceback
import cv2
import mediapipe as mp
import numpy as np
import sys
import os
from PIL import ImageFont, ImageDraw, Image
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.infer import predict_emotion
from python.recorder import EmotionLogger

def run_face_detection():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    logger = EmotionLogger()
    emotion_counter = Counter()

    # ✅ Mac 전용 한글 폰트
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    font = ImageFont.truetype(font_path, 28)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다.")
            return

        print("🎥 얼굴 인식 및 감정 분석 시작 (ESC 키로 종료)")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다.")
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            image_pil = Image.fromarray(image_bgr)
            draw = ImageDraw.Draw(image_pil)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image_bgr, detection)

                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image_bgr.shape
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)

                    face_crop = image_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                    if face_crop.size == 0 or face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                        print("⚠️ 감정 추론 건너뜀: 얼굴 영역이 너무 작음")
                        logger.log("unclassified")
                        continue

                    try:
                        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        emotion = predict_emotion(face_gray)
                        korean_label = emotion_to_korean(emotion)

                        # ✅ 감정 카운트 누적
                        emotion_counter[korean_label] += 1

                        # ✅ 얼굴 위에 감정 표시
                        draw.text((x1, y1 - 30), korean_label, font=font, fill=(0, 255, 0))
                        logger.log(emotion)

                    except Exception as e:
                        print(f"⚠️ 감정 추론 실패: {e}")
                        traceback.print_exc()
                        logger.log("unclassified")

            # ✅ 화면 상단에 감정 카운트 표시
            count_text = "   ".join([f"{label} {count}" for label, count in emotion_counter.items()])
            draw.text((10, 10), count_text, font=font, fill=(255, 255, 255))

            image_bgr = np.array(image_pil)
            cv2.imshow("Face Detection + Emotion", image_bgr)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def emotion_to_korean(emotion: str) -> str:
    return {
        "angry": "화남",
        "disgust": "역겨움",
        "fear": "공포",
        "happy": "기쁨",
        "neutral": "중립",
        "sad": "슬픔",
        "surprise": "놀람",
        "unclassified": "분류불가"
    }.get(emotion, "알수없음")

if __name__ == "__main__":
    run_face_detection()
