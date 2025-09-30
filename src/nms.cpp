#include <knr/nms.h>

#include <opencv2/opencv.hpp>

std::expected<cv::Mat, std::string> non_maximum_suppression(const cv::Mat &mag) {
    if (mag.type() != CV_8UC1)
        return std::unexpected("Input magnitude matrix is not 8UC1");

    cv::Mat nms_mag{};
    nms_mag.create(mag.size(), mag.type());

    // Stuff

    return nms_mag;
}
