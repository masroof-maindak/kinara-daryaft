#include <knr/args.h>
#include <knr/gauss.h>
#include <knr/hysteresis.h>
#include <knr/io.h>
#include <knr/nms.h>
#include <knr/utils.h>

#include <opencv2/opencv.hpp>

#include <expected>

std::expected<cv::Mat, std::string> canny_edge_detector(const std::string &img_name, const cv::Mat &img,
                                                        const ArgConfig &args, bool save_intermediates = false) {
    // --- G + Gx/Gy ---
    const int filt_size{compute_filter_size(args.sigma, args.T)};

    const auto gaussian_filt_expected{generate_gaussian_filter(filt_size, args.sigma)};
    if (!gaussian_filt_expected.has_value())
        return std::unexpected{"Failed to generate gaussian filter: {}" + gaussian_filt_expected.error()};

    const cv::Mat filt{gaussian_filt_expected.value()};

    const auto part_der_result_expected{compute_gaussian_derivatives(filt, args.sigma)};
    if (!part_der_result_expected.has_value())
        return std::unexpected{"Failed to partial derivatives of Gaussian: {}" + part_der_result_expected.error()};

    const auto [gx, gy]{part_der_result_expected.value()};

    // --- Fx/Fy ---
    const cv::Mat img_padded{pad_image(img, gx.rows / 2)};

    const auto fx_expected{convolve_through_image(img_padded, gx)};
    if (!fx_expected.has_value())
        return std::unexpected{"Failed to compute image fx: {}" + fx_expected.error()};

    const cv::Mat fx{fx_expected.value()};

    const auto fy_expected{convolve_through_image(img_padded, gy)};

    if (!fy_expected.has_value())
        return std::unexpected{"Failed to compute image fy: {}" + fy_expected.error()};

    const cv::Mat fy{fy_expected.value()};

    // --- Gradient Direction ---
    const auto grad_dir_expected{compute_gradient_direction(fx, fy)};
    if (!grad_dir_expected.has_value())
        return std::unexpected{"Failed to generate gradient directions: " + grad_dir_expected.error()};

    const cv::Mat grad_dir{grad_dir_expected.value()};

    // --- Gradient Magnitude + Save ---
    const auto grad_mag_expected{compute_gradient_magnitude(fx, fy)};
    if (!grad_mag_expected.has_value())
        return std::unexpected{"Failed to generate gradient magections: " + grad_mag_expected.error()};

    const cv::Mat grad_mag{grad_mag_expected.value()};

    if (save_intermediates) {
        const auto mag_sv_expected{save_image(grad_mag, args.out_dir, img_name, "magnitude", args.sigma)};
        if (!mag_sv_expected.has_value())
            return std::unexpected{"Failed to save image grad_mag: {}" + mag_sv_expected.error()};
    }

    // --- Non-Maximum Suppresion + Save ---
    const auto nms_mag_expected{non_maximum_suppression(grad_mag, grad_dir)};
    if (!nms_mag_expected.has_value())
        return std::unexpected{"Failed to generate nms mat: " + nms_mag_expected.error()};

    const cv::Mat nms_mag{nms_mag_expected.value()};

    if (save_intermediates) {
        const auto nms_sv_expected{save_image(nms_mag, args.out_dir, img_name, "nms", args.sigma)};
        if (!nms_sv_expected.has_value())
            return std::unexpected{"Failed to save image nms: {}" + nms_sv_expected.error()};
    }

    // --- Hysteresis Thresholding + Save ---
    const auto thresholded_mag_expected{apply_hysteresis(nms_mag, args.low_threshold, args.high_threshold)};
    if (!thresholded_mag_expected.has_value())
        return std::unexpected{"Failed to apply hysteresis thresholding: " + thresholded_mag_expected.error()};

    return thresholded_mag_expected.value();
}
