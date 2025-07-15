#include "emotion_utils.hpp"
#include <iostream>
#include <unordered_map>
#include <string>
#include <sstream>
#include <mutex>

namespace {
    std::unordered_map<std::string, int> emotion_counts;
    std::mutex emotion_mutex;
}

void accumulate_emotion(const std::string& emotion) {
    std::lock_guard<std::mutex> lock(emotion_mutex);
    emotion_counts[emotion]++;
}

std::string get_emotion_stats() {
    std::lock_guard<std::mutex> lock(emotion_mutex);
    std::ostringstream oss;
    for (const auto& pair : emotion_counts) {
        oss << pair.first << ": " << pair.second << "\n"; 
    }
    return oss.str();
}

void reset_emotions() {
    std::lock_guard<std::mutex> lock(emotion_mutex);
    emotion_counts.clear();
}
