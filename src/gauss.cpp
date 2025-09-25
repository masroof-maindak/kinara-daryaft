#include <cmath>
#include <knr/gauss.h>

#include <cstdint>
#include <math.h>

int compute_filter_size(float sigma, float T) {
    int size_half{static_cast<int>(sqrt(-log(T)) * 2 * sigma * sigma)};
    return 2 * size_half + 1;
}

std::pair<cv::Mat, float> generate_gaussian_filter(const int filter_size, const float sigma) {
    cv::Mat filt_f{};
    filt_f.create(filter_size, filter_size, CV_32FC1);

    for (int y = 0; y < filter_size; y++) {
        float *filt_row = filt_f.ptr<float>(y);
        for (int x = 0; x < filter_size; x++)
            filt_row[x] = exp(-(x * x + y * y) / (2 * sigma * sigma));
    }

    float min{}, max{}, scale_factor{};

    // TODO: find minimum and maximum from filt_f to determine scaling factor

    // TODO: copy over into filt_u8

    cv::Mat filt_u8{};
    filt_u8.create(filter_size, filter_size, CV_8UC1);

    for (int y = 0; y < filter_size; y++) {

        std::uint8_t *filt_u8_row = filt_u8.ptr<std::uint8_t>(y);
        const float *filt_f_row   = filt_f.ptr<float>(y);

        for (int x = 0; x < filter_size; x++)
            filt_u8_row[x] = filt_f_row[x] * scale_factor;
    }

    return {filt_u8, scale_factor};
}

std::pair<cv::Mat, cv::Mat> compute_partial_derivatives(const cv::Mat &gaussian_filt) {
    cv::Mat gx{};
    cv::Mat gy{};
    gx.create(gaussian_filt.size(), CV_8UC1);
    gy.create(gaussian_filt.size(), CV_8UC1);

    // TODO: derivatives?

    return {gx, gy};
}
