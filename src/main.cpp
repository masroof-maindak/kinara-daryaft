#include <knr/args.h>
#include <knr/gauss.h>
#include <knr/io.h>
#include <knr/utils.h>

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <print>
#include <stdlib.h>

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
    auto img_expected = load_image(args.img_path);
    if (!img_expected.has_value()) {
        std::println(stderr, "Failed to load image: {}", args_expected.error());
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

    auto part_der_result_expected{compute_partial_derivatives(filt, args.sigma)};
    if (!part_der_result_expected.has_value()) {
        std::println(stderr, "Failed to partial derivatives of Gaussian: {}", part_der_result_expected.error());
        return EXIT_FAILURE;
    }
    const auto [gx, gy]{part_der_result_expected.value()};

    // --- Fx/Fy + Save ---
    const cv::Mat img_padded{pad_image(img, gx.rows / 2)};

    auto fx_expected{convolve_through_image(img_padded, gx)};
    if (!fx_expected.has_value()) {
        std::println(stderr, "Failed to compute image fx: {}", fx_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat fx{fx_expected.value()};

    /*
     * NOTE: Fx & Yy can't be saved because they comprise 32 bit integers. This is a given because convolution
     * w/ Gx & Gy can result in negative values.
     */

    // auto fx_save_res_expected = save_image(fx, args.out_dir, img_name, "fx", args.sigma);
    // if (!fx_save_res_expected.has_value()) {
    //     std::println(stderr, "Failed to save image fx: {}", fx_save_res_expected.error());
    //     return EXIT_FAILURE;
    // }

    const auto fy_expected{convolve_through_image(img_padded, gy)};
    if (!fy_expected.has_value()) {
        std::println(stderr, "Failed to compute image fy: {}", fy_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat fy{fy_expected.value()};

    // auto fy_save_res_expected = save_image(fy, args.out_dir, img_name, "fy", args.sigma);
    // if (!fy_save_res_expected.has_value()) {
    //     std::println(stderr, "Failed to save image fy: {}", fy_save_res_expected.error());
    //     return EXIT_FAILURE;
    // }

    // --- Gradient Direction ---
    auto grad_dir_expected{compute_gradient_direction(fx, fy)};
    if (!grad_dir_expected.has_value()) {
        std::println(stderr, "Failed to generate gradient directions: ", grad_dir_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat grad_dir{grad_dir_expected.value()};

    // --- Gradient Magnitude + Save ---
    auto grad_mag_expected{compute_gradient_magnitude(fx, fy)};
    if (!grad_mag_expected.has_value()) {
        std::println(stderr, "Failed to generate gradient magections: ", grad_mag_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat grad_mag{grad_mag_expected.value()};

    auto mag_save_res_expected = save_image(grad_mag, args.out_dir, img_name, "magnitude", args.sigma);
    if (!mag_save_res_expected.has_value()) {
        std::println(stderr, "Failed to save image fy: {}", mag_save_res_expected.error());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
