#include <knr/args.h>

std::expected<ArgConfig, std::string> parse_args(int argc, char *argv[]) {
    argparse::ArgumentParser prog("knr", "v2025-09-17a", argparse::default_arguments::help);

    ArgConfig args{};

    prog.add_argument("-i").required().help("specify the input image").store_into(args.img_path);

    prog.add_argument("-o", "--output-dir").required().help("specify the output dir").store_into(args.out_dir);

    prog.add_argument("-s", "--sigma")
        .help("specify value of sigma to be used for determining the Gaussian filter's size")
        .default_value(1.0f)
        .store_into(args.sigma) // Broken when it's at the bottom, because why
                                // wouldn't it be.
        .scan<'g', float>();

    prog.add_argument("-T")
        .help("specify value of T to be used for determining the Gaussian filter's size")
        .default_value(0.3f)
        .store_into(args.T)
        .scan<'g', float>();

    prog.add_argument("-lt", "--low-threshold")
        .help("specify the low threshold to be used hysteresis thresholding")
        .default_value(2)
        .scan<'i', int>()
        .store_into(args.low_threshold);

    prog.add_argument("-ht", "--high-threshold")
        .help("specify the high threshold to be used hysteresis thresholding")
        .default_value(6)
        .scan<'i', int>()
        .store_into(args.high_threshold);

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception &err) {
        auto errmsg = std::format("{}\n\n{}", err.what(), prog.usage());
        return std::unexpected(errmsg);
    }

    if (prog.is_used("-T")) {
        float f{prog.get<float>("T")};
        if (f < 0 || f > 1)
            return std::unexpected(std::format("T must lie between 0 and 1: {}", f));
    }

    if (prog.is_used("-lt")) {
        int i{prog.get<int>("-lt")};
        if (i <= 0)
            return std::unexpected(std::format("low threshold must be +ve: {}", i));
    }

    if (prog.is_used("-ht")) {
        int i{prog.get<int>("-ht")};
        if (i <= 0)
            return std::unexpected(std::format("high threshold must be +ve: {}", i));
    }

    if (prog.is_used("-s")) {
        float f{prog.get<float>("-s")};
        if (f < 0.5)
            return std::unexpected(std::format("Sigma can't be lower than 0.5: {}", f));
    }

    return args;
}
