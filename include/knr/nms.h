#ifndef NMS_H
#define NMS_H

#include <opencv2/core/mat.hpp>

#include <expected>
#include <string>

namespace kd {

// Takes 2x 8UC1
// Returns 8UC1
std::expected<cv::Mat, std::string> non_maximum_suppression(const cv::Mat &grad_mag, const cv::Mat &grad_dir);

} // namespace kd

#endif // NMS_H
