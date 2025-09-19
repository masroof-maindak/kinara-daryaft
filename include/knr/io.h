#ifndef IO_H
#define IO_H

#include <opencv2/opencv.hpp>

#include <expected>

std::expected<cv::Mat, std::string> load_image(const std::string &path);

std::expected<void, std::string> save_image(const cv::Mat &img, const std::string &out_dir, const std::string &name,
                                            const std::string &phase, const float sigma);

#endif // IO_H
