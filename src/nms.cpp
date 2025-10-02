#include <knr/nms.h>
#include <knr/utils.h>

#include <opencv2/opencv.hpp>

#include <cstdint>

std::expected<cv::Mat, std::string> non_maximum_suppression(const cv::Mat &grad_mag, const cv::Mat &grad_dir) {
    using enum GradientDir;

    if (grad_mag.type() != CV_8UC1)
        return std::unexpected("Input magnitude matrix is not 8UC1");

    if (grad_dir.type() != CV_8UC1)
        return std::unexpected("Input direction matrix is not 8UC1");

    cv::Mat nms_mag{grad_mag.size(), grad_mag.type(), cv::Scalar::all(0)};

    const int rows{nms_mag.rows};
    const int cols{nms_mag.cols};

    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            const auto dir{grad_dir.at<std::uint8_t>(y, x)};

            Px n1{}, n2{};

            switch (static_cast<GradientDir>(dir)) {
            case E_W:
                n1 = {y, x - 1};
                n2 = {y, x + 1};
                break;
            case NE_SW:
                n1 = {y - 1, x - 1};
                n2 = {y + 1, x + 1};
                break;
            case N_S:
                n1 = {y - 1, x};
                n2 = {y + 1, x};
                break;
            case NW_SE:
                n1 = {y - 1, x + 1};
                n2 = {y + 1, x - 1};
                break;
            default:
                return std::unexpected("Gradient direction matrix had unexpected value: {}");
            }

            auto mag_at_px = [grad_mag](std::pair<int, int> p) -> std::uint8_t {
                return grad_mag.at<std::uint8_t>(p.first, p.second);
            };

            const auto curr_mag{mag_at_px({y, x})};

            if (curr_mag > mag_at_px(n1) && curr_mag > mag_at_px(n2))
                nms_mag.at<std::uint8_t>(y, x) = curr_mag;
        }
    }

    return nms_mag;
}
