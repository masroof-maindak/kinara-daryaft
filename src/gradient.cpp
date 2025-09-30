#include <knr/gradient.h>
#include <knr/utils.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>

std::expected<cv::Mat, std::string> compute_gradient_direction(const cv::Mat &fx, const cv::Mat &fy) {
    if (fx.size() != fy.size())
        return std::unexpected(std::format("fx.size != fy.size ; {}x{} & {}x{}", fx.rows, fx.cols, fy.rows, fy.cols));

    if (fx.type() != CV_32SC1 || fy.type() != CV_32SC1)
        return std::unexpected("Invalid type for fx or fy; expected CV_32SC1");

    cv::Mat dir{};
    dir.create(fx.size(), CV_8UC1);

    const int rows{fx.rows};
    const int cols{fx.cols};
    constexpr float rad_to_deg{180 / std::numbers::pi_v<float>};

    using enum GradientDir;

    for (int y = 0; y < rows; y++) {

        const auto fx_row{fx.ptr<std::int32_t>(y)};
        const auto fy_row{fy.ptr<std::int32_t>(y)};
        auto dir_row{dir.ptr<std::uint8_t>(y)};

        for (int x = 0; x < cols; x++) {
            float t{atan2f(fy_row[x], fx_row[x]) * rad_to_deg + 180};

            dir_row[x] = ((t >= 0 && t < 22.5) || (t >= 157.5 && t < 202.5) || (t >= 337.5 && t <= 360)) ? +E_W
                         : ((t >= 22.5 && t < 67.5) || (t >= 202.5 && t < 247.5))                        ? +NE_SW
                         : ((t >= 67.5 && t < 112.5) || (t >= 247.5 && t < 292.5))                       ? +N_S
                         : ((t >= 112.5 && t < 157.5) || (t >= 292.5 && t < 337.5))                      ? +NW_SE
                                                                                                         : +Invalid;
            if (dir_row[x] == 255)
                return std::unexpected(std::format("Invalid theta encountered {}", t));
        }
    }

    return dir;
}

std::expected<cv::Mat, std::string> compute_gradient_magnitude(const cv::Mat &fx, const cv::Mat &fy) {
    if (fx.size() != fy.size())
        return std::unexpected(std::format("fx.size != fy.size ; {}x{} & {}x{}", fx.rows, fx.cols, fy.rows, fy.cols));

    if (fx.type() != CV_32SC1 || fy.type() != CV_32SC1)
        return std::unexpected("Invalid type for fx or fy; expected CV_32SC1");

    cv::Mat mag{};
    cv::Mat temp{};
    mag.create(fx.size(), CV_8UC1);
    temp.create(fx.size(), CV_32FC1);

    const int rows{fx.rows};
    const int cols{fx.cols};
    const float scale_factor{256};

    // Determine Magnitude
    for (int y = 0; y < rows; y++) {

        const auto fx_row{fx.ptr<std::int32_t>(y)};
        const auto fy_row{fy.ptr<std::int32_t>(y)};
        auto temp_row{temp.ptr<float>(y)};

        for (int x = 0; x < cols; x++)
            temp_row[x] = sqrtf(fx_row[x] * fx_row[x] + fy_row[x] * fy_row[x]) / scale_factor;
    }

    // Scale magnitude b/w 0 and 255
    const auto [min_p, max_p]{std::ranges::minmax_element(
        reinterpret_cast<float *>(temp.data), reinterpret_cast<float *>(temp.data + temp.elemSize() * temp.total()))};

    const float min{*min_p};
    const float max{*max_p};

    if (min == max) {
        memset(mag.data, 0, mag.elemSize() * mag.total());
        return mag;
    }

    for (int y = 0; y < rows; y++) {

        const auto temp_row{temp.ptr<float>(y)};
        auto *mag_row{mag.ptr<std::uint8_t>(y)};

        for (int x = 0; x < cols; x++)
            mag_row[x] = static_cast<std::uint8_t>(std::round((255 * (temp_row[x] - min)) / (max - min)));
    }

    return mag;
}
