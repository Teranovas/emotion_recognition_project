import cpp_module

from python.face_detector import run_face_detection

if __name__ == "__main__":
    run_face_detection()
    print(dir(cpp_module))
