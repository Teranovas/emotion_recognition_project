import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

# ✅ 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# ✅ 로그 디렉토리 경로
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

# ✅ 가장 최신 로그 파일 선택
def get_latest_log_file():
    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".csv")]
    if not log_files:
        raise FileNotFoundError("logs 디렉토리에 CSV 파일이 없습니다.")
    log_files.sort(reverse=True)
    return os.path.join(LOG_DIR, log_files[0])

# ✅ 감정 라벨을 한글로 변환
def emotion_to_korean(emotion):
    return {
        "angry": "화남",
        "disgust": "역겨움",
        "fear": "공포",
        "happy": "기쁨",
        "neutral": "중립",
        "sad": "슬픔",
        "surprise": "놀람",
        "unclassified": "미분류"
    }.get(emotion, "알수없음")

# ✅ 시각화 함수
def visualize_emotion_log(csv_path):
    df = pd.read_csv(csv_path)

    if df.empty:
        print("❗ 로그 파일이 비어 있습니다.")
        return

    # ✅ 한글 감정 라벨로 변환
    df["emotion_kr"] = df["emotion"].map(emotion_to_korean)

    # ✅ 시간 변환
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # ✅ 선 그래프: 시간에 따른 감정 변화
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["emotion_kr"], marker='o', linestyle='-', alpha=0.7)
    plt.title("시간에 따른 감정 변화")
    plt.xlabel("시간")
    plt.ylabel("감정")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ✅ 막대그래프: 감정별 등장 횟수
    plt.figure(figsize=(8, 5))
    df["emotion_kr"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("감정별 등장 횟수")
    plt.xlabel("감정")
    plt.ylabel("횟수")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        latest_csv = get_latest_log_file()
        print(f"📄 최신 로그 파일: {latest_csv}")
        visualize_emotion_log(latest_csv)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
