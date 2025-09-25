#ifndef ARGS_H
#define ARGS_H

#include <argparse/argparse.hpp>

#include <expected>
#include <string>

struct ArgConfig {
    float sigma;
    float T;
    int low_threshold;
    int high_threshold;
    std::string img_path;
    std::string out_dir;
};

std::expected<ArgConfig, std::string> parse_args(int argc, char *argv[]);

#endif // ARGS_H
