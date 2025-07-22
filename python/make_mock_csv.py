# make_mock_csv.py
import csv
from datetime import datetime, timedelta

start_time = datetime(2025, 7, 22, 12, 0)
emotions = ['기쁨', '기쁨', '슬픔', '기쁨', '놀람', '슬픔', '화남', '기쁨', '중립']

with open("logs/mock_emotion_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "emotion"])
    for i, e in enumerate(emotions):
        time = start_time + timedelta(minutes=i)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), e])
