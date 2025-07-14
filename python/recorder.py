import csv
import os
from datetime import datetime

class EmotionLogger:
    def __init__(self, log_dir="logs", filename_prefix="emotion_log"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.csv")
        self.header_written = False

    def log(self, emotion: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.file_path, mode='a', newline='') as f:
            writer = csv.writer(f)

            if not self.header_written:
                writer.writerow(["timestamp", "emotion"])
                self.header_written = True

            writer.writerow([timestamp, emotion])
