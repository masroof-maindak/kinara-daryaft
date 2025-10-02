#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/mat.hpp>

#include <cstdint>

cv::Mat pad_image(const cv::Mat &img, const int padding);

/*
 * This is wholly unnecessary but I just wanted to play around w/ scoped enums and their goofy little operator overload.
 */
enum class GradientDir : std::uint8_t {
    E_W     = 0, // East ↔ West
    NE_SW   = 1, // Northeast ↔ Southwest
    N_S     = 2, // North ↔ South
    NW_SE   = 3, // Northwest ↔ Southeast
    Invalid = 255
};

std::uint8_t operator+(const GradientDir gd);

using Px = std::pair<int, int>;

#endif // UTILS_H
