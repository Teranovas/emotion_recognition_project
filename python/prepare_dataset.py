import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

FER_CSV_PATH = "data/raw/fer2013.csv"
OUTPUT_DIR = "data/processed/FER2013_images"

emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

def convert_csv_to_images():
    print("📁 FER2013 CSV가 존재합니다.")

    try:
        df = pd.read_csv(
            FER_CSV_PATH,
            sep=",",
            engine="python",
            on_bad_lines="skip"
        )
    except Exception as e:
        print(f"❌ CSV 로딩 실패: {e}")
        return

    print(f"✅ CSV 컬럼 목록: {list(df.columns)}")  # <--- 여기에 컬럼명 출력 추가

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for usage in ["Training", "PublicTest", "PrivateTest"]:
        usage_dir = os.path.join(OUTPUT_DIR, usage)
        os.makedirs(usage_dir, exist_ok=True)

        if "Usage" not in df.columns:
            print("❌ 'Usage' 컬럼이 없습니다. CSV 파일 구조를 확인하세요.")
            return

        subset = df[df["Usage"] == usage]
        print(f"▶ {usage}: {len(subset)}개 이미지 생성 중...")

        for i, row in tqdm(subset.iterrows(), total=len(subset)):
            try:
                emotion = emotion_labels[int(row["emotion"])]
                pixels = np.fromstring(row["pixels"], sep=" ")
                image = pixels.reshape(48, 48).astype(np.uint8)
                image_path = os.path.join(usage_dir, f"{emotion}_{i}.png")
                Image.fromarray(image).save(image_path)
            except Exception as e:
                print(f"⚠️ 이미지 생성 실패: {e}")

if __name__ == "__main__":
    convert_csv_to_images()
