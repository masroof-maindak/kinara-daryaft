#include <knr/hysteresis.h>

#include <cstring>
#include <format>

std::expected<cv::Mat, std::string> apply_hysteresis(const cv::Mat &mag, const int low_thresh, const int high_thresh) {
    if (low_thresh < 0 || high_thresh < 0)
        return std::unexpected(std::format("Negative threshold is invalid: {} and {}", low_thresh, high_thresh));

    if (mag.type() != CV_8UC1)
        return std::unexpected("Expected intensity matrix to be of type CV_8UC1");

    cv::Mat temp_mag{mag.clone()};
    cv::Mat thresh_mag{mag.size(), CV_8UC1, cv::Scalar::all(0)};

    // Set borders to 0 in original
    const auto temp_rows{temp_mag.rows};
    const auto temp_cols{temp_mag.cols};
    const auto bytes_per_row{temp_mag.elemSize() * temp_cols};
    std::memset(temp_mag.ptr<std::uint8_t>(0), 0, bytes_per_row);
    std::memset(temp_mag.ptr<std::uint8_t>(temp_rows - 1), 0, bytes_per_row);
    for (int y = 1; y < temp_rows - 1; y++) {
        temp_mag.at<std::uint8_t>(y, 0)             = 0;
        temp_mag.at<std::uint8_t>(y, temp_cols - 1) = 0;
    }

    // TODO: Recurse when first pixel is found w/ intensity over high_thresh

    return thresh_mag;
}
