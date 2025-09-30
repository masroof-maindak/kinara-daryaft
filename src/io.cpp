#include <knr/io.h>

#include <filesystem>

std::expected<cv::Mat, std::string> load_image(const std::string &path) {
    cv::Mat img{cv::imread(path, cv::IMREAD_GRAYSCALE)};

    if (img.empty())
        return std::unexpected("Failed to read/parse image: " + path);

    return img;
}

std::expected<void, std::string> save_image(const cv::Mat &img, const std::string &out_dir, const std::string &img_name,
                                            const std::string &phase, const float sigma) {
    if (!std::filesystem::exists(out_dir)) {
        std::error_code e;
        if (!std::filesystem::create_directories(out_dir, e))
            return std::unexpected("Failed to create directory: " + e.message());
    }

    auto fname{std::format("{}/{}_{}_{}.jpg", out_dir, img_name, phase, sigma)};
    // TODO: handle imwrite errors
    cv::imwrite(fname, img);

    return {};
}
