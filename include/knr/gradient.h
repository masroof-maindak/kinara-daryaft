#ifndef GRADIENT_H
#define GRADIENT_H

#include <opencv2/core/mat.hpp>

#include <expected>
#include <string>

namespace kd {

// Takes 2x 32SC1 (fx, fy)
// returns the QUANTIZED gradient direction as an 8UC1 matrix
std::expected<cv::Mat, std::string> compute_gradient_direction(const cv::Mat &fx, const cv::Mat &fy);

// Takes 2x 32SC1 (fx, fy)
// returns 8UC1
std::expected<cv::Mat, std::string> compute_gradient_magnitude(const cv::Mat &fx, const cv::Mat &fy);

} // namespace kd

#endif // GRADIENT_H
