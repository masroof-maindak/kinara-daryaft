#ifndef NMS_H
#define NMS_H

#include <opencv2/core/mat.hpp>

#include <expected>
#include <string>

std::expected<cv::Mat, std::string> nms();

#endif // NMS_H
