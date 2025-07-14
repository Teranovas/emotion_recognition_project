import cv2
import mediapipe as mp

def run_face_detection():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # MediaPipe 얼굴 감지 객체 초기화
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다.")
            return

        print("🎥 얼굴 인식 시작 (ESC 키로 종료)")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다.")
                break

            # BGR → RGB 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 얼굴 감지
            results = face_detection.process(image)

            # 결과 시각화용: 다시 RGB → BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 얼굴이 감지되면 사각형 그리기
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)

            cv2.imshow("Face Detection", image)

            # ESC 누르면 종료
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_detection()
