#include <knr/args.h>
#include <knr/gauss.h>
#include <knr/io.h>

#include <opencv2/opencv.hpp>

#include <print>

int main(int argc, char *argv[]) {
    auto args_expected = parse_args(argc, argv);
    if (!args_expected.has_value()) {
        std::println(stderr, "Failed to parse args: {}", args_expected.error());
        return 1;
    }
    ArgConfig args{args_expected.value()};

    auto img_expected = load_image(args.img_path);
    if (!img_expected.has_value()) {
        std::println(stderr, "Failed to load image: {}", args_expected.error());
        return 1;
    }
    const auto img{img_expected.value()};

    const int filt_size = compute_filter_size(args.sigma, args.T);

    const auto [filt, scale_factor] = generate_gaussian_filter(filt_size, args.sigma);

    const auto [gx, gy] = compute_partial_derivatives(filt);

    const auto fx = convolve_through_image(img, gx);

    const auto fy = convolve_through_image(img, gy);

    return 0;
}
