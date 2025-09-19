#include "knr/io.h"
#include <knr/args.h>
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
    auto img{img_expected.value()};

    return 0;
}
