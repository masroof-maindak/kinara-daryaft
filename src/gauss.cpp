#include <knr/gauss.h>

#include <math.h>
#include <stdint.h>

int compute_filter_size(float sigma, float T) {
    const int half_size{static_cast<int>(std::round(sqrt(-std::logf(T)) * 2 * sigma * sigma))};
    return 2 * half_size + 1;
}

std::pair<cv::Mat, float> generate_gaussian_filter(const int filter_size, const float sigma) {
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

    filt /= sum;

    return {filt, sum};
}

std::pair<cv::Mat, cv::Mat> compute_partial_derivatives(const cv::Mat &filt_f, const float sigma) {
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

    std::cout << gx_f << "\n\n" << gy_f << "\n\n";

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

    return {gx_i16, gy_i16};
}

std::pair<cv::Mat, cv::Mat> compute_partial_derivatives(const cv::Mat &gaussian_filt) {
    cv::Mat gx{};
    cv::Mat gy{};
    gx.create(gaussian_filt.size(), CV_8UC1);
    gy.create(gaussian_filt.size(), CV_8UC1);

    // TODO: derivatives?

    return {gx, gy};
}
