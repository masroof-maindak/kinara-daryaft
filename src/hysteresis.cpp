#include <knr/hysteresis.h>
#include <knr/utils.h>

#include <array>
#include <format>
#include <queue>

std::expected<cv::Mat, std::string> kd::apply_hysteresis(const cv::Mat &mag, const int low_thresh,
                                                         const int high_thresh) {
    if (low_thresh < 0 || high_thresh < 0 || low_thresh > 255 || high_thresh > 255)
        return std::unexpected(std::format("Threshold not in (0,255]: {} or {}", low_thresh, high_thresh));

    if (high_thresh <= low_thresh)
        return std::unexpected(std::format("tH must exceed lT: {} and {}", high_thresh, low_thresh));

    if (mag.type() != CV_8UC1)
        return std::unexpected("Expected intensity matrix to be of type CV_8UC1");

    cv::Mat padded_mag{pad_image(mag, 1)};
    cv::Mat thresh_mag{mag.size(), CV_8UC1, cv::Scalar::all(0)};
    auto visited{std::vector<std::vector<bool>>(padded_mag.rows, std::vector<bool>(padded_mag.cols, false))};

    auto mag_at_px = [padded_mag](std::pair<int, int> p) -> std::uint8_t {
        return padded_mag.at<std::uint8_t>(p.first, p.second);
    };

    for (int y = 1; y < mag.rows - 1; y++) {
        for (int x = 1; x < mag.cols - 1; x++) {

            const Px curr_px{y, x};

            if (visited[curr_px.first][curr_px.second])
                continue;

            if (mag_at_px(curr_px) > high_thresh) {
                std::queue<Px> to_visit({curr_px});
                visited[curr_px.first][curr_px.second] = true;

                while (!to_visit.empty()) {
                    const auto px = to_visit.front();
                    to_visit.pop();
                    const auto curr_mag = mag_at_px(px);

                    if (curr_mag >= low_thresh)
                        thresh_mag.at<std::uint8_t>(px.first - 1, px.second - 1) = curr_mag;
                    else
                        continue;

                    std::array<Px, 8> neighbours{{{px.first - 1, px.second - 1},
                                                  {px.first - 1, px.second},
                                                  {px.first - 1, px.second + 1},
                                                  {px.first, px.second - 1},
                                                  {px.first, px.second + 1},
                                                  {px.first + 1, px.second - 1},
                                                  {px.first + 1, px.second},
                                                  {px.first + 1, px.second + 1}}};

                    for (const auto n : neighbours) {
                        if (!visited[n.first][n.second]) {
                            to_visit.push(n);
                            visited[n.first][n.second] = true;
                        }
                    }
                }
            }
        }
    }

    return thresh_mag;
}
