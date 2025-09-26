#include <knr/gauss.h>

#include <math.h>
#include <numeric>
#include <stdint.h>

int compute_filter_size(float sigma, float T) {
    const int half_size{static_cast<int>(std::round(sqrt(-std::logf(T)) * 2 * sigma * sigma))};
    return 2 * half_size + 1;
}

std::expected<cv::Mat, std::string> generate_gaussian_filter(const int filter_size, const float sigma) {
    // TODO: check if sigma is in bounds
    // TODO: check if filter size is positive and odd

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

std::expected<std::pair<cv::Mat, cv::Mat>, std::string> compute_partial_derivatives(const cv::Mat &filt_f,
                                                                                    const float sigma) {
    // TODO: validate filt_f's type
    // TODO: validate sigma's bounds

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

    // TODO: check if fogd is larger than image.

    const int half_size{fogd.rows / 2};

    cv::Mat f_part{};
    f_part.create(img_padded.rows - (fogd.rows - 1), img_padded.cols - (fogd.cols - 1), CV_8UC1);

    std::vector<uint8_t> patch{};
    patch.reserve(fogd.rows * fogd.cols);

    std::span<std::int16_t> fogd_flat{reinterpret_cast<std::int16_t *>(fogd.data),
                                      reinterpret_cast<std::int16_t *>(fogd.data + fogd.elemSize() * fogd.total())};

    for (int y = half_size; y < img_padded.rows - half_size; y++) {

        std::uint8_t *f_row = f_part.ptr<std::uint8_t>(y - half_size);

        for (int x = half_size; x < img_padded.cols - half_size; x++) {

            // Populate patch
            for (int yy = y - half_size; yy <= y + half_size; yy++) {
                const std::uint8_t *padded_row = img_padded.ptr<std::uint8_t>(yy);
                for (int xx = x - half_size; xx <= x + half_size; xx++)
                    patch.emplace_back(padded_row[xx]);
            }

            const int dot_prod{std::inner_product(patch.begin(), patch.end(), fogd_flat.begin(), 0)};
            // CHECK: Narrowing conversion. Take max?
            f_row[x - half_size] = dot_prod;
            patch.clear();
        }
    }

    return f_part;
}

cv::Mat compute_gradient_direction(const cv::Mat &fx, const cv::Mat &fy) {
    // TODO: ensure fx and fy are equal size
    // TODO: ensure fx and fy are type CV_8UC1

    cv::Mat dir{};
    dir.create(fx.size(), CV_32FC1);

    const int rows{fx.rows};
    const int cols{fx.cols};

    // CHECK: div by 0
    for (int i = 0; i < rows * cols; i++)
        dir.data[i] = atanf(static_cast<float>(fy.data[i]) / fx.data[i]);

    return dir;
}

cv::Mat compute_gradient_magnitude(const cv::Mat &fx, const cv::Mat &fy) {
    // TODO: ensure fx and fy are equal size
    // TODO: ensure fx and fy are type CV_8UC1

    cv::Mat mag{};
    mag.create(fx.size(), CV_8UC1);

    const int rows{fx.rows};
    const int cols{fx.cols};
    const float scale_factor{256};

    for (int i = 0; i < rows * cols; i++)
        mag.data[i] = static_cast<int>(std::round(sqrt(pow(fx.data[i], 2) + pow(fy.data[i], 2) / scale_factor)));

    // TODO: normalize b/w 0 and 100
    // TODO: normalize patch-by-patch instead (to make result lighting invariant)

    return mag;
}
