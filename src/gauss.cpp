#include <knr/gauss.h>

#include <math.h>
#include <stdint.h>

int compute_filter_size(float sigma, float T) {
    int half_size{static_cast<int>(sqrt(-log(T)) * 2 * sigma * sigma)};
    return 2 * half_size + 1;
}

std::pair<cv::Mat, float> generate_gaussian_filter(const int filter_size, const float sigma) {
    cv::Mat filt_f{};
    filt_f.create(filter_size, filter_size, CV_32FC1);

    const float two_sigma_sq{2 * sigma * sigma};
    const int half_size{filter_size / 2};

    double sum{};
    for (int y = 0; y < filter_size; y++) {
        float *filt_row = filt_f.ptr<float>(y);
        for (int x = 0; x < filter_size; x++) {
            int xx{std::abs(x - half_size)};
            int yy{std::abs(y - half_size)};
            filt_row[x] = (1 / std::numbers::pi * two_sigma_sq) * exp(-(xx * xx + yy * yy) / two_sigma_sq);
            sum += filt_row[x];
        }
    }

    filt_f /= sum;

    cv::Mat filt_u8{};
    filt_u8.create(filter_size, filter_size, CV_8UC1);

    for (int y = 0; y < filter_size; y++) {
        const float *filt_f_row   = filt_f.ptr<float>(y);
        std::uint8_t *filt_u8_row = filt_u8.ptr<std::uint8_t>(y);
        for (int x = 0; x < filter_size; x++)
            filt_u8_row[x] = static_cast<std::uint8_t>(filt_f_row[x] * 255);
    }

    return {filt_u8, sum};
}

std::pair<cv::Mat, cv::Mat> compute_partial_derivatives(const cv::Mat &gaussian_filt) {
    cv::Mat gx{};
    cv::Mat gy{};
    gx.create(gaussian_filt.size(), CV_8UC1);
    gy.create(gaussian_filt.size(), CV_8UC1);

    // TODO: derivatives?

    return {gx, gy};
}
