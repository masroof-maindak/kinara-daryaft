#include <knr/nms.h>
#include <knr/utils.h>

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <optional>

std::expected<cv::Mat, std::string> non_maximum_suppression(const cv::Mat &grad_mag, const cv::Mat &grad_dir) {
    if (grad_mag.type() != CV_8UC1)
        return std::unexpected("Input magnitude matrix is not 8UC1");

    if (grad_dir.type() != CV_8UC1)
        return std::unexpected("Input direction matrix is not 8UC1");

    using enum GradientDir;
    using NeighbourPx = std::optional<std::pair<int, int>>;

    cv::Mat nms_mag{grad_mag.size(), grad_mag.type(), cv::Scalar::all(0)};

    const int rows{nms_mag.rows};
    const int cols{nms_mag.cols};

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            const auto dir{grad_dir.at<std::uint8_t>(y, x)};

            NeighbourPx n1{}, n2{};

            switch (static_cast<GradientDir>(dir)) {
            case E_W:
                if (x > 0) // left
                    n1 = {y, x - 1};

                if (x + 1 < cols) // right
                    n2 = {y, x + 1};
                break;
            case NE_SW:
                if (y > 0 && x > 0) // top-left
                    n1 = {y - 1, x - 1};

                if (y + 1 < rows && x + 1 < cols) // bottom-right
                    n2 = {y + 1, x + 1};

                break;
            case N_S:
                if (y > 0) // up
                    n1 = {y - 1, x};

                if (y + 1 < rows) // down
                    n2 = {y + 1, x};

                break;
            case NW_SE:
                if (y > 0 && x + 1 < cols) // top-right
                    n1 = {y - 1, x + 1};

                if (y + 1 < rows && x > 0) // bottom-left
                    n2 = {y + 1, x - 1};

                break;
            default:
                return std::unexpected("Gradient direction matrix had unexpected value: {}");
            }

            auto mag = [grad_mag](std::pair<int, int> p) -> std::uint8_t {
                return grad_mag.at<std::uint8_t>(p.first, p.second);
            };

            if (n1.has_value() && n2.has_value()) {
                const auto curr_mag{mag({y, x})};

                if (curr_mag > mag(*n1) && curr_mag > mag(*n2))
                    nms_mag.at<std::uint8_t>(y, x) = curr_mag;
            }
        }
    }

    return nms_mag;
}
