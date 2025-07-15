#include <pybind11/pybind11.h>
#include "emotion_utils.hpp"

namespace py = pybind11;

// Python → C++ 함수 노출
PYBIND11_MODULE(cpp_module, m) {
    m.def("process_emotion", &process_emotion, "감정을 처리하는 C++ 함수");
}