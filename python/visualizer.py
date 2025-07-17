import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
