#include "args.h"

#include <knr/canny.h>
#include <knr/io.h>
#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <filesystem>
#include <print>

int main(int argc, char *argv[]) {
    // --- Config ---
    const auto args_expected = parse_args(argc, argv);
    if (!args_expected.has_value()) {
        std::println(stderr, "Failed to parse args: {}", args_expected.error());
        return EXIT_FAILURE;
    }
    const ArgConfig args{args_expected.value()};

    const std::string img_name{std::filesystem::path{args.img_path}.stem()};

    // --- Load Image ---
    const auto img_expected{load_image(args.img_path)};
    if (!img_expected.has_value()) {
        std::println(stderr, "Failed to load image: {}", img_expected.error());
        return EXIT_FAILURE;
    }
    const cv::Mat img{img_expected.value()};

    // --- Canny ---
    const auto thresh_mag_expected{canny_edge_detector
        img_name, img, {args.sigma, args.T, args.low_threshold, args.high_threshold, args.out_dir}, true)};
    if (!thresh_mag_expected.has_value()) {
        std::println(stderr, "Failed to run canny: {}", thresh_mag_expected.error());
        return EXIT_FAILURE;
    }

    const cv::Mat thresh_mag{thresh_mag_expected.value()};

    const auto hyst_phase_name{std::format("hysteresis_{}_{}", args.low_threshold, args.high_threshold)};
    const auto thresh_mag_save_expected{save_image(thresh_mag, args.out_dir, img_name, hyst_phase_name, args.sigma)};
    if (!thresh_mag_save_expected.has_value()) {
        std::println(stderr, "Failed to save edge detection image: {}", thresh_mag_save_expected.error());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
