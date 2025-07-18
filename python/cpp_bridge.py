import cpp_module

def send_emotion_to_cpp(emotion: str):

    cpp_module.send_emotion(emotion)

def get_emotion_stats_from_cpp() -> str:
    return cpp_module.get_emotion_stats()

def reset_cpp_stats():
    cpp_module.reset_emotions()
