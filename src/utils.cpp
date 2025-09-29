#include <knr/utils.h>

#include <opencv2/core/types.hpp>

#include <cstring>

cv::Mat pad_image(const cv::Mat &img, const int padding) {
    cv::Mat padded{img.rows + 2 * padding, img.cols + 2 * padding, img.type(), cv::Scalar(0)};

    for (int y = 0; y < img.rows; y++) {
        const std::uint8_t *original_row = img.ptr<std::uint8_t>(y);
        std::uint8_t *padded_row         = padded.ptr<std::uint8_t>(y + padding);
        std::memcpy(padded_row + padding, original_row, img.cols);
    }

    return padded;
}

std::uint8_t operator+(GradientDir gd) { return std::to_underlying(gd); }
