#ifndef GAUSS_H
#define GAUSS_H

#include <opencv2/core/mat.hpp>

#include <expected>
#include <string>
#include <utility>

int compute_filter_size(float sigma, float T);

// Generates a normalized gaussian filter of floats
// Also returns the scale factor so the user can 'scale down' later i.e during magnitude calculation.
// G (x, y) = exp (-(x^2 + y^2)/(2*sigma^2))
std::expected<cv::Mat, std::string> generate_gaussian_filter(const int filter_size, const float sigma);

// Computes Gx and Gy given a matrix of floats representing a Gaussian filter
// Gx and Gy hold 16 bit integers
std::expected<std::pair<cv::Mat, cv::Mat>, std::string> compute_gaussian_derivatives(const cv::Mat &filt_f,
                                                                                     const float sigma);

// Convolves a first-order Gaussian derivative (Gx or Gy) through a (padded!) source image
// img is 8UC1 (grayscale)
// fogd is 16SC1 (16 bit signed int)
// returns 32SC1 image (fx/fy) of edge detections as per input fogd (gx/gy)
std::expected<cv::Mat, std::string> convolve_through_image(const cv::Mat &img_padded, const cv::Mat &fogd);

// Takes 2x 32SC1 (fx, fy)
// returns the QUANTIZED gradient direction as an 8UC1 matrix
std::expected<cv::Mat, std::string> compute_gradient_direction(const cv::Mat &fx, const cv::Mat &fy);

// Takes 2x 32SC1 (fx, fy)
// returns 8UC1
std::expected<cv::Mat, std::string> compute_gradient_magnitude(const cv::Mat &fx, const cv::Mat &fy);

#endif // GAUSS_H
