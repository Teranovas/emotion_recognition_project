#include <iostream>
#include <string>

std::string process_emotion(const std::string& emotion) {
    std::cout << "[C++] 감정 처리 중: " << emotion << std::endl;
    return "C++가 처리한 감정: " + emotion;
}
