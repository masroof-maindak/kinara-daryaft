#include <knr/hysteresis.h>
#include <knr/utils.h>

#include <array>
#include <chrono>
#include <format>
#include <print>
#include <queue>

using hrs = std::chrono::high_resolution_clock;

std::expected<cv::Mat, std::string> apply_hysteresis(const cv::Mat &mag, const int low_thresh, const int high_thresh) {
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

    /*
     * TODO: Optimise performance.
     *
     * First identify what is the most costly part of the function:
     * - `visited` accesses? Try replacing it w/ a boolean map.
     * - While loop? Try a recursive solution and pray the compiler somehow optimises it to the moon and back.
     * - Non-cache-friendly memory accesses? Maybe store high-threshold pixels during a first pass and iterate through
     * rows on subsequent passes whilst maintaining a map of coordinates to know when you're 'above' or 'below' a
     * high-threshold pixel.
     *
     * ---
     *
     * NOTE - Observations:
     *
     * - The for loop for the neighbours takes 98% of the time of its parent while.
     *   - But only 10% of the time of the whole 2d loop
     * - the first visited.contains() call eats up 86% of the time of the whole matrix' loop, so let's get rid of that
     * first
     * - After replacing the unordered set w/ a boolean matrix, I'm still seeing 46% of the 2d loop's time spent on the
     * first visisted check, surprisingly, and I doubt a bitfield would fare any better.
     */

    hrs::duration while_time{};
    hrs::duration vis_time{};
    hrs::duration neighbours_time{};

    auto s_total = hrs::now();
    for (int y = 1; y < mag.rows - 1; y++) {
        for (int x = 1; x < mag.cols - 1; x++) {

            const Px curr_px{y, x};

            auto s_vis_cont = hrs::now();
            if (visited[curr_px.first][curr_px.second])
                continue;
            auto e_vis_cont = hrs::now();
            vis_time += (e_vis_cont - s_vis_cont);

            if (mag_at_px(curr_px) > high_thresh) {
                std::queue<Px> to_visit({curr_px});
                visited[curr_px.first][curr_px.second] = true;

                auto s_while = hrs::now();

                while (!to_visit.empty()) {
                    const auto px = to_visit.front();
                    to_visit.pop();

                    const auto curr_mag = mag_at_px(px);

                    if (curr_mag < low_thresh)
                        continue;

                    thresh_mag.at<std::uint8_t>(px.first - 1, px.second - 1) = curr_mag;

                    std::array<Px, 8> neighbours{{{px.first - 1, px.second - 1},
                                                  {px.first - 1, px.second},
                                                  {px.first - 1, px.second + 1},
                                                  {px.first, px.second - 1},
                                                  {px.first, px.second + 1},
                                                  {px.first + 1, px.second - 1},
                                                  {px.first + 1, px.second},
                                                  {px.first + 1, px.second + 1}}};

                    auto s_neighbours = hrs::now();

                    for (const auto n : neighbours) {
                        if (!visited[n.first][n.second]) {
                            to_visit.push(n);
                            visited[n.first][n.second] = true;
                        }
                    }

                    auto e_neighbours = hrs::now();
                    neighbours_time += (e_neighbours - s_neighbours);
                }

                auto e_while = hrs::now();
                while_time += (e_while - s_while);
            }
        }
    }
    auto e_total = hrs::now();

    std::println("Total: {}", (e_total - s_total));
    std::println("Is visited?: {}", vis_time);
    std::println("While: {}", while_time);
    std::println("Neighbours: {}", neighbours_time);

    return thresh_mag;
}
