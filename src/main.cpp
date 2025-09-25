#include <knr/args.h>
#include <knr/gauss.h>
#include <knr/io.h>

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <print>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    auto args_expected = parse_args(argc, argv);
    if (!args_expected.has_value()) {
        std::println(stderr, "Failed to parse args: {}", args_expected.error());
        return EXIT_FAILURE;
    }
    ArgConfig args{args_expected.value()};

    std::string img_name = std::filesystem::path{args.img_path}.stem();

    auto img_expected = load_image(args.img_path);
    if (!img_expected.has_value()) {
        std::println(stderr, "Failed to load image: {}", args_expected.error());
        return EXIT_FAILURE;
    }
    const auto img{img_expected.value()};

    const int filt_size = compute_filter_size(args.sigma, args.T);

    const auto [filt, scale_factor] = generate_gaussian_filter(filt_size, args.sigma);

    const auto [gx, gy] = compute_partial_derivatives(filt, args.sigma);

    auto fx_expected{convolve_through_image(img_padded, gx)};
    if (!fx_expected.has_value()) {
        std::println(stderr, "Failed to compute image fx: {}", fx_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat fx{fx_expected.value()};

    auto fx_save_res_expected = save_image(fx, args.out_dir, img_name, "fx", args.sigma);
    if (!fx_save_res_expected.has_value()) {
        std::println(stderr, "Failed to save image fx: {}", fx_save_res_expected.error());
        return EXIT_FAILURE;
    }

    const auto fy_expected{convolve_through_image(img_padded, gy)};
    if (!fy_expected.has_value()) {
        std::println(stderr, "Failed to compute image fy: {}", fy_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat fy{fy_expected.value()};

    auto fy_save_res_expected = save_image(fy, args.out_dir, img_name, "fy", args.sigma);
    if (!fy_save_res_expected.has_value()) {
        std::println(stderr, "Failed to save image fy: {}", fy_save_res_expected.error());
        return EXIT_FAILURE;
    }

    // TODO: gradient direction/magnitude

    return EXIT_SUCCESS;
}
