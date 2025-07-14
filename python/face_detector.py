import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import mediapipe as mp
import numpy as np
from model.infer import predict_emotion  # 감정 추론 함수 불러오기

def run_face_detection():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

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
                    # 얼굴 박스 시각화
                    mp_drawing.draw_detection(image_bgr, detection)

                    # 상대 좌표 → 절대 픽셀 좌표로 변환
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image_bgr.shape
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)

                    # 얼굴 영역 자르기 (안전하게 clipping)
                    face_crop = image_bgr[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                    if face_crop.size == 0:
                        continue

                    # 감정 추론
                    try:
                        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        emotion = predict_emotion(face_gray)
                        # 감정 텍스트 오버레이
                        cv2.putText(image_bgr, f'{emotion}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"⚠️ 감정 추론 실패: {e}")

            cv2.imshow("Face Detection + Emotion", image_bgr)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_detection()
