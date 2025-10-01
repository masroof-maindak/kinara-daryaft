#include <knr/args.h>
#include <knr/gauss.h>
#include <knr/io.h>
#include <knr/nms.h>
#include <knr/utils.h>

#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <filesystem>
#include <print>

int main(int argc, char *argv[]) {
    // --- Config ---
    auto args_expected = parse_args(argc, argv);
    if (!args_expected.has_value()) {
        std::println(stderr, "Failed to parse args: {}", args_expected.error());
        return EXIT_FAILURE;
    }
    ArgConfig args{args_expected.value()};

    const std::string img_name{std::filesystem::path{args.img_path}.stem()};

    // --- Load Image ---
    auto img_expected{load_image(args.img_path)};
    if (!img_expected.has_value()) {
        std::println(stderr, "Failed to load image: {}", img_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat img{img_expected.value()};

    // --- G + Gx/Gy ---
    const int filt_size{compute_filter_size(args.sigma, args.T)};
    const auto gaussian_filt_expected{generate_gaussian_filter(filt_size, args.sigma)};
    if (!gaussian_filt_expected.has_value()) {
        std::println(stderr, "Failed to generate gaussian filter: {}", gaussian_filt_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat filt{gaussian_filt_expected.value()};

    auto part_der_result_expected{compute_gaussian_derivatives(filt, args.sigma)};
    if (!part_der_result_expected.has_value()) {
        std::println(stderr, "Failed to partial derivatives of Gaussian: {}", part_der_result_expected.error());
        return EXIT_FAILURE;
    }
    const auto [gx, gy]{part_der_result_expected.value()};

    // --- Fx/Fy ---
    const cv::Mat img_padded{pad_image(img, gx.rows / 2)};

    /*
     * NOTE: Fx & Yy can't be saved because they comprise 32 bit integers. This is a given because convolution
     * w/ Gx & Gy can result in negative values.
     */

    auto fx_expected{convolve_through_image(img_padded, gx)};
    if (!fx_expected.has_value()) {
        std::println(stderr, "Failed to compute image fx: {}", fx_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat fx{fx_expected.value()};

    const auto fy_expected{convolve_through_image(img_padded, gy)};
    if (!fy_expected.has_value()) {
        std::println(stderr, "Failed to compute image fy: {}", fy_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat fy{fy_expected.value()};

    // --- Gradient Direction ---
    auto grad_dir_expected{compute_gradient_direction(fx, fy)};
    if (!grad_dir_expected.has_value()) {
        std::println(stderr, "Failed to generate gradient directions: ", grad_dir_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat grad_dir{grad_dir_expected.value()};

    /*
     * NOTE: Quantized gradient directions also have no reason to be saved because the only values they contain are 0,
     * 1, 2, 3.
     */

    // --- Gradient Magnitude + Save ---
    auto grad_mag_expected{compute_gradient_magnitude(fx, fy)};
    if (!grad_mag_expected.has_value()) {
        std::println(stderr, "Failed to generate gradient magections: ", grad_mag_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat grad_mag{grad_mag_expected.value()};

    auto mag_save_res_expected{save_image(grad_mag, args.out_dir, img_name, "magnitude", args.sigma)};
    if (!mag_save_res_expected.has_value()) {
        std::println(stderr, "Failed to save image grad_mag: {}", mag_save_res_expected.error());
        return EXIT_FAILURE;
    }

    // --- Non-Maximum Suppresion + Save ---
    auto nms_mag_expected{non_maximum_suppression(grad_mag, grad_dir)};
    if (!nms_mag_expected.has_value()) {
        std::println(stderr, "Failed to generate nms mat: ", nms_mag_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat nms_mag{nms_mag_expected.value()};

    auto nms_save_res_expected{save_image(nms_mag, args.out_dir, img_name, "nms", args.sigma)};
    if (!nms_save_res_expected.has_value()) {
        std::println(stderr, "Failed to save image nms: {}", nms_save_res_expected.error());
        return EXIT_FAILURE;
    }

    // TODO: Hysteresis Thresholding

    // CHECK: move value out of std::expected?

    return EXIT_SUCCESS;
}
