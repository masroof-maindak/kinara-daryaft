#ifndef IO_H
#define IO_H

#include <opencv2/opencv.hpp>

#include <expected>

namespace kd {

std::expected<cv::Mat, std::string> load_image(const std::string &path);

std::expected<void, std::string> save_image(const cv::Mat &img, const std::string &out_dir, const std::string &name,
                                            const std::string &phase, const float sigma);

} // namespace kd

#endif // IO_H
