#include <knr/args.h>

#include <opencv2/opencv.hpp>

#include <print>

int main(int argc, char *argv[]) {
    auto args_expected = parse_args(argc, argv);
    if (!args_expected.has_value()) {
        std::println(stderr, "Failed to parse args: {}", args_expected.error());
        return 1;
    }
    ArgConfig args{args_expected.value()};

    return 0;
}
