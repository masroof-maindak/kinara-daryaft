#include <knr/nms.h>

#include <opencv2/opencv.hpp>

#include <cstdint>

std::expected<cv::Mat, std::string> non_maximum_suppression(const cv::Mat &grad_mag, const cv::Mat &grad_dir) {
    if (grad_mag.type() != CV_8UC1)
        return std::unexpected("Input magnitude matrix is not 8UC1");

    if (grad_dir.type() != CV_8UC1)
        return std::unexpected("Input direction matrix is not 8UC1");

    cv::Mat nms_mag{};
    nms_mag.create(grad_mag.size(), grad_mag.type());

    const int rows{nms_mag.rows};
    const int cols{nms_mag.cols};

    // CHECK: leave 1 row/col's worth of padding

    for (int y = 0; y < rows; y++) {

        const auto *mag_row{grad_mag.ptr<std::uint8_t>(y)};
        const auto *dir_row{grad_dir.ptr<std::uint8_t>(y)};
        auto *nms_row{nms_mag.ptr<std::uint8_t>(y)};

        for (int x = 0; x < cols; y++) {
            /*
             * TODO: For every nearby edge in the edge map (intensity matrix) that shares the same direction, if the
             * current pixel in consideration has a higher intensity than them, reduce the others to zero.
             *
             * CHECK: What classifies as a 'vicinity' i.e how to pick 'nearby' edges. Could I just add a black 1 px-wide
             * border (we already have pad_image in utils.cpp) and then check all 8 neighbouring pixels?
             */
        }
    }

    return nms_mag;
}
