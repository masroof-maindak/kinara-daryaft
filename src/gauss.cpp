#include <knr/gauss.h>

#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstdint>
#include <format>
#include <numeric>
#include <span>

int compute_filter_size(float sigma, float T) {
    const int half_size{static_cast<int>(std::round(sqrt(-std::logf(T)) * 2 * sigma * sigma))};
    return 2 * half_size + 1;
}

std::expected<cv::Mat, std::string> generate_gaussian_filter(const int filter_size, const float sigma) {
    if (sigma < 0.5)
        return std::unexpected(std::format("Small sigma, expected sigma > 0.5: {}", sigma));

    if (filter_size < 0 || filter_size % 2 == 0)
        return std::unexpected(std::format("Filter size should be +ve & odd: {}", filter_size));

    cv::Mat filt{};
    filt.create(filter_size, filter_size, CV_32FC1);

    const float two_sigma_sq{2 * sigma * sigma};
    const int half_size{filter_size / 2};

    float sum{};
    for (int y = 0; y < filter_size; y++) {

        float *filt_row = filt.ptr<float>(y);
        const int dy{y - half_size};

        for (int x = 0; x < filter_size; x++) {
            const int dx{x - half_size};
            filt_row[x] = exp(-(dx * dx + dy * dy) / two_sigma_sq);
            sum += filt_row[x];
        }
    }

    // Normalize
    filt /= sum;

    return filt;
}

std::expected<std::pair<cv::Mat, cv::Mat>, std::string> compute_gaussian_derivatives(const cv::Mat &filt_f,
                                                                                     const float sigma) {
    if (sigma < 0.5)
        return std::unexpected(std::format("Small sigma, expected sigma > 0.5: {}", sigma));

    if (filt_f.type() != CV_32FC1)
        return std::unexpected("Filter was not of type CV_32FC1");

    cv::Mat gx_f{};
    cv::Mat gy_f{};
    gx_f.create(filt_f.size(), CV_32FC1);
    gy_f.create(filt_f.size(), CV_32FC1);

    const int filter_size{filt_f.rows};
    const int half_size{filter_size / 2};
    const float inv_sigma_sq{1 / sigma * sigma};

    // Solve Gx & Gy (floats)
    for (int y = 0; y < filter_size; y++) {

        const int dy{y - half_size};
        const float *filt_f_row = filt_f.ptr<float>(y);
        float *gx_f_row         = gx_f.ptr<float>(y);
        float *gy_f_row         = gy_f.ptr<float>(y);

        for (int x = 0; x < filter_size; x++) {
            const int dx{x - half_size};
            gx_f_row[x] = -dx * inv_sigma_sq * filt_f_row[x];
            gy_f_row[x] = -dy * inv_sigma_sq * filt_f_row[x];
        }
    }

    // Convert Gx & Gy to int
    cv::Mat gx_i16{};
    cv::Mat gy_i16{};
    gx_i16.create(filt_f.size(), CV_16SC1);
    gy_i16.create(filt_f.size(), CV_16SC1);

    const float scale_factor{256};

    for (int y = 0; y < filter_size; y++) {
        const float *gx_f_row    = gx_f.ptr<float>(y);
        const float *gy_f_row    = gy_f.ptr<float>(y);
        std::int16_t *gx_i16_row = gx_i16.ptr<std::int16_t>(y);
        std::int16_t *gy_i16_row = gy_i16.ptr<std::int16_t>(y);

        for (int x = 0; x < filter_size; x++) {
            gx_i16_row[x] = static_cast<std::int16_t>(std::round(gx_f_row[x] * scale_factor));
            gy_i16_row[x] = static_cast<std::int16_t>(std::round(gy_f_row[x] * scale_factor));
        }
    }

    return std::pair{gx_i16, gy_i16};
}

std::expected<cv::Mat, std::string> convolve_through_image(const cv::Mat &img_padded, const cv::Mat &fogd) {
    if (img_padded.type() != CV_8UC1)
        return std::unexpected("Unexpected image type; require CV_8UC1 (grayscale).");

    if (fogd.type() != CV_16SC1)
        return std::unexpected("Unexpected partial derivative type; require CV_16SC1.");

    // NOTE: accounts for padding!
    const int fogd_size{fogd.rows};
    if (fogd_size > img_padded.rows || fogd_size > img_padded.cols)
        return std::unexpected(
            std::format("FOGD size {} can't exceed image res {}x{}.", fogd_size, img_padded.rows, img_padded.cols));

    const int half_size{fogd.rows / 2};

    cv::Mat f_part{};
    f_part.create(img_padded.rows - (fogd.rows - 1), img_padded.cols - (fogd.cols - 1), CV_32SC1);

    std::vector<uint8_t> patch{};
    patch.reserve(fogd.rows * fogd.cols);

    std::span<std::int16_t> fogd_flat{reinterpret_cast<std::int16_t *>(fogd.data),
                                      reinterpret_cast<std::int16_t *>(fogd.data + fogd.elemSize() * fogd.total())};

    for (int y = half_size; y < img_padded.rows - half_size; y++) {

        std::int32_t *f_row = f_part.ptr<std::int32_t>(y - half_size);

        for (int x = half_size; x < img_padded.cols - half_size; x++) {

            // Populate patch
            for (int yy = y - half_size; yy <= y + half_size; yy++) {
                const std::uint8_t *padded_row = img_padded.ptr<std::uint8_t>(yy);
                for (int xx = x - half_size; xx <= x + half_size; xx++)
                    patch.emplace_back(padded_row[xx]);
            }

            const int32_t dot_prod{std::inner_product(patch.begin(), patch.end(), fogd_flat.begin(), 0)};
            f_row[x - half_size] = dot_prod;
            patch.clear();
        }
    }

    return f_part;
}
