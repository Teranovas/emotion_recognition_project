#ifndef EMOTION_UTILS_HPP
#define EMOTION_UTILS_HPP

#include <string>

void accumulate_emotion(const std::string& emotion);
std::string get_emotion_stats();
void reset_emotions();

#endif
