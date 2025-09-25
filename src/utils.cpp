#include <knr/utils.h>
#include <opencv2/core/types.hpp>

cv::Mat pad_image(const cv::Mat &img, int padding) {
    cv::Mat padded{img.rows + 2 * padding, img.cols + 2 * padding, img.type(), cv::Scalar::ones()};

    for (int y = 0; y < img.rows; y++) {
        const std::uint8_t *original_row = img.ptr<std::uint8_t>(y);
        std::uint8_t *padded_row         = padded.ptr<std::uint8_t>(y + padding);

        for (int x = 0; x < img.cols; x++)
            padded_row[x + padding] = original_row[x];
    }

    return padded;
}
