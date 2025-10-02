#include <knr/hysteresis.h>
#include <knr/utils.h>

#include <array>
#include <format>
#include <queue>
#include <unordered_set>

std::expected<cv::Mat, std::string> apply_hysteresis(const cv::Mat &mag, const int low_thresh, const int high_thresh) {
    if (low_thresh < 0 || high_thresh < 0)
        return std::unexpected(std::format("Negative threshold is invalid: {} or {}", low_thresh, high_thresh));

    if (high_thresh <= low_thresh)
        return std::unexpected(std::format("tH must exceed lT: {} and {}", high_thresh, low_thresh));

    if (mag.type() != CV_8UC1)
        return std::unexpected("Expected intensity matrix to be of type CV_8UC1");

    cv::Mat temp_mag{pad_image(mag, 1)};
    cv::Mat thresh_mag{mag.size(), CV_8UC1, cv::Scalar::all(0)};

    std::unordered_set<Px, HashPx> visited{};

    auto mag_at_px = [temp_mag](std::pair<int, int> p) -> std::uint8_t {
        return temp_mag.at<std::uint8_t>(p.first, p.second);
    };

    for (int y = 1; y < mag.rows - 1; y++) {
        for (int x = 1; x < mag.cols - 1; x++) {

            std::pair<int, int> curr_px{y, x};

            if (visited.contains(curr_px))
                continue;

            if (mag_at_px(curr_px) > high_thresh) {
                std::queue<Px> to_visit({curr_px});

                while (!to_visit.empty()) {
                    const auto px = to_visit.front();
                    to_visit.pop();
                    visited.insert(px);

                    const auto curr_mag = mag_at_px(px);

                    if (curr_mag < low_thresh)
                        continue;

                    thresh_mag.at<std::uint8_t>(px.first - 1, px.second - 1) = curr_mag;

                    const auto px_y{px.first};
                    const auto px_x{px.second};

                    std::array<Px, 8> neighbours{{{px.first - 1, px.second - 1},
                                                  {px.first - 1, px.second},
                                                  {px.first - 1, px.second + 1},
                                                  {px.first, px.second - 1},
                                                  {px.first, px.second + 1},
                                                  {px.first + 1, px.second - 1},
                                                  {px.first + 1, px.second},
                                                  {px.first + 1, px.second + 1}}};

                    for (const auto n : neighbours)
                        if (!visited.contains(n))
                            to_visit.push(n);
                }
            }
        }
    }

    return thresh_mag;
}
