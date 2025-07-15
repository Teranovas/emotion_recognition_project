#include <pybind11/pybind11.h>
#include "emotion_utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cpp_module, m) {
    m.def("send_emotion", &accumulate_emotion);     // ✅ 감정 누적
    m.def("get_emotion_stats", &get_emotion_stats); // ✅ 감정 통계
    m.def("reset_emotions", &reset_emotions);       // ✅ 통계 초기화
}
