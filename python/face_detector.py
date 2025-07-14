import traceback  # 상단에 추가
import cv2
import mediapipe as mp
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.infer import predict_emotion
from python.recorder import EmotionLogger

def run_face_detection():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    logger = EmotionLogger()

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        cap = cv2.VideoCapture(0)

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

                    # 유효성 검사
                    if face_crop.size == 0 or face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                        print("⚠️ 감정 추론 건너뜀: 얼굴 영역이 너무 작음")
                        logger.log("unclassified")
                        continue

                    try:
                        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        emotion = predict_emotion(face_gray)

                        cv2.putText(image_bgr, f'{emotion}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                        logger.log(emotion)

                    except Exception as e:
                        print(f"⚠️ 감정 추론 실패: {e}")
                        traceback.print_exc()
                        logger.log("unclassified")

            cv2.imshow("Face Detection + Emotion", image_bgr)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_detection()
