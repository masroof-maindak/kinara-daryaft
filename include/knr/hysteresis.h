#ifndef HYSTERESIS_H
#define HYSTERESIS_H

#include <opencv2/opencv.hpp>

#include <expected>
#include <string>

// Takes a grayscale 8UC1 matrix and two thresholds
// Returns a grayscale 8UC1 matrix
std::expected<cv::Mat, std::string> apply_hysteresis(const cv::Mat &mag, const int low_thresh, const int high_thresh);

#endif // HYSTERESIS_H
