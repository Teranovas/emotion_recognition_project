import traceback
import cv2
import mediapipe as mp
import numpy as np
import sys
import os
from PIL import ImageFont, ImageDraw, Image
from collections import Counter

# 상위 디렉토리 모듈 불러오기 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.infer import predict_emotion
from python.recorder import EmotionLogger
from python.cpp_bridge import send_emotion_to_cpp, get_emotion_stats_from_cpp, reset_cpp_stats
from python.visualizer import plot_emotion_bar_chart

def open_available_camera(priority_order=[1, 0, 2, 3]):
    for index in priority_order:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"✅ 카메라 {index}번 장치 사용됨")
            return cap
        else:
            cap.release()
    print("❌ 사용 가능한 카메라 장치를 찾을 수 없습니다.")
    return None

def run_face_detection():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    logger = EmotionLogger()
    emotion_counter = Counter()

    # ✅ Mac용 한글 폰트 경로
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    font = ImageFont.truetype(font_path, 28)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        cap = open_available_camera()
        if cap is None:
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

            # PIL 이미지 변환 (한글 라벨용)
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

                        # ✅ 감정 카운트
                        emotion_counter[korean_label] += 1

                        # ✅ 얼굴 위에 감정 라벨 표시 (한글)
                        draw.text((x1, y1 - 30), korean_label, font=font, fill=(0, 255, 0))

                        # ✅ 감정 로깅
                        logger.log(emotion)

                        # ✅ C++로 감정 전달
                        send_emotion_to_cpp(emotion)

                    except Exception as e:
                        print(f"⚠️ 감정 추론 실패: {e}")
                        traceback.print_exc()
                        logger.log("unclassified")

            # ✅ 상단 감정 카운트 표시
            count_text = "   ".join([f"{label} {count}" for label, count in emotion_counter.items()])
            draw.text((10, 10), count_text, font=font, fill=(255, 255, 255))

            image_bgr = np.array(image_pil)
            cv2.imshow("Face Detection + Emotion", image_bgr)

            # ✅ 키 입력 처리
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                print("\n📊 [C++] 감정 통계")
                stats_str = get_emotion_stats_from_cpp()
                print(stats_str, end="") 
                stats_dict = parse_emotion_stats(stats_str)
                plot_emotion_bar_chart(stats_dict)
            elif key == ord('r'):
                reset_cpp_stats()
                print("🔁 감정 통계 초기화됨")
            elif key == ord('f'):
                emotion_filter = input("🎯 필터링할 감정을 한글로 입력하세요 (예: 기쁨): ").strip()
                if emotion_filter:
                    print(f"🔎 [{emotion_filter}] 감정 필터링 모드 활성화됨")
                    # 필터링 로직은 따로 구현되어 있다고 가정

        cap.release()
        cv2.destroyAllWindows()

# 감정 코드 → 한글 변환 함수
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

def parse_emotion_stats(stats_str: str) -> dict:
    stats = {}
    lines = stats_str.strip().splitlines()
    for line in lines:
        if ':' in line:
            emotion, count = line.split(':')
            stats[emotion.strip()] = int(count.strip())
    return stats

if __name__ == "__main__":
    run_face_detection()
