#ifndef CANNY_H
#define CANNY_H

#include <opencv2/opencv.hpp>

#include <expected>

struct CannyCfg {
    float sigma;
    float T;
    int low_threshold;
    int high_threshold;
    std::string out_dir;
};

std::expected<cv::Mat, std::string> canny_edge_detector(const std::string &img_name, const cv::Mat &img,
                                                        const CannyCfg &args, bool save_intermediates = false);

#endif // CANNY_H
