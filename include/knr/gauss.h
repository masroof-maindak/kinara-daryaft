#ifndef GAUSS_H
#define GAUSS_H

#include <opencv2/opencv.hpp>

#include <utility>

int compute_filter_size(float sigma, float T);

// Generates a gaussian filter of 8 bit unsigned integers (as opposed to floats that add up to 1)
// Also returns the scale factor so the user can 'scale down' later i.e during magnitude calculation.
// G (x, y) = exp (-(x^2 + y^2)/(2*sigma^2))
std::pair<cv::Mat, float> generate_gaussian_filter(const int filter_size, const float sigma);

// Computes Gx and Gy given a matrix representing a Gaussian filter
std::pair<cv::Mat, cv::Mat> compute_partial_derivatives(const cv::Mat &gaussian_filt);

// Convolves a first-order Gaussian derivative (Gx or Gy) through a source image
cv::Mat convolve_through_image(const cv::Mat &img, const cv::Mat &fogd);

cv::Mat compute_gradient_direction(const cv::Mat &fx, const cv::Mat &fy);

cv::Mat compute_gradient_magnitude(const cv::Mat &fx, const cv::Mat &fy);

#endif // GAUSS_H
