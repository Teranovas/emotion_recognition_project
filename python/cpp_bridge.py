import sys
import os

# 빌드된 .so/.dylib 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))

try:
    import cpp_module
except ImportError as e:
    print("❌ C++ 모듈 불러오기 실패: cpp_module.so가 build에 없습니다.")
    raise e

def send_emotion_to_cpp(emotion: str):
    cpp_module.process_emotion(emotion)
