import matplotlib.pyplot as plt

def plot_emotion_bar_chart(stats_str: str):
    # C++에서 온 문자열을 줄 단위로 분리
    lines = [line.strip() for line in stats_str.strip().split('\n') if line]
    labels = []
    counts = []

    for line in lines:
        if ':' in line:
            label, count = line.split(':')
            labels.append(label.strip())
            counts.append(int(count.strip()))

    # 바 그래프 그리기
    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts)
    plt.title("감정 통계 (C++ 처리 결과)")
    plt.xlabel("감정")
    plt.ylabel("횟수")
    plt.tight_layout()
    plt.show()
