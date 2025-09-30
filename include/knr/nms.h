#ifndef NMS_H
#define NMS_H

#include <opencv2/core/mat.hpp>

#include <expected>
#include <string>

std::expected<cv::Mat, std::string> non_maximum_suppression(const cv::Mat &mag);

#endif // NMS_H
