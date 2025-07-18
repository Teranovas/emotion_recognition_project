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
