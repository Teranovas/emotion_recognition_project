import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def plot_emotion_bar_chart(emotion_counts: dict):
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    fontprop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fontprop.get_name()

    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())

    plt.figure(figsize=(8, 6))
    plt.bar(emotions, counts)
    plt.title("감정 분포 (C++ 연산 결과)")
    plt.xlabel("감정")
    plt.ylabel("빈도")
    plt.tight_layout()
    plt.show()

def plot_emotion_trend(csv_path: str):
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    fontprop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fontprop.get_name()

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 감정별로 10초 단위 빈도수 리샘플링
    resampled = df.resample('10S').emotion.value_counts().unstack().fillna(0)

    plt.figure(figsize=(10, 6))
    for emotion in resampled.columns:
        plt.plot(resampled.index, resampled[emotion], marker='o', label=emotion)

    plt.title("시간 흐름에 따른 감정 변화")
    plt.xlabel("시간")
    plt.ylabel("감정 빈도")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def save_sample_bar_chart():
    sample_counts = {
        "기쁨": 4,
        "슬픔": 2,
        "놀람": 1,
        "화남": 1,
        "중립": 1
    }
    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    fontprop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fontprop.get_name()

    plt.figure(figsize=(8, 6))
    plt.bar(sample_counts.keys(), sample_counts.values())
    plt.title("감정 분포 예시")
    plt.xlabel("감정")
    plt.ylabel("빈도")
    plt.tight_layout()
    plt.savefig("assets/sample_bar.png")
    print("✅ sample_bar.png 저장 완료")

def save_sample_trend_chart():
    import pandas as pd
    df = pd.read_csv("logs/mock_emotion_log.csv", parse_dates=["timestamp"])
    emotion_map = {'기쁨': 0, '슬픔': 1, '놀람': 2, '화남': 3, '중립': 4}
    df["emotion_code"] = df["emotion"].map(emotion_map)

    font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    fontprop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = fontprop.get_name()

    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp"], df["emotion_code"], marker='o')
    plt.yticks(list(emotion_map.values()), list(emotion_map.keys()))
    plt.title("감정 변화 트렌드 예시")
    plt.xlabel("시간")
    plt.ylabel("감정")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assets/sample_trend.png")
    print("✅ sample_trend.png 저장 완료")
