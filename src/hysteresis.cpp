#include <knr/hysteresis.h>
#include <knr/utils.h>

#include <array>
#include <chrono>
#include <format>
#include <print>
#include <queue>

using hrs      = std::chrono::high_resolution_clock;
using Duration = hrs::duration;
using Nanos    = std::chrono::nanoseconds;

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

    Duration while_time{};
    Duration mat_rw_time{};
    Duration vis_time{};
    Duration neighbours_time{};

    auto start_total = hrs::now();

    for (int y = 1; y < mag.rows - 1; y++) {
        for (int x = 1; x < mag.cols - 1; x++) {

            const Px curr_px{y, x};

            auto start_visited_check = hrs::now();
            if (visited[curr_px.first][curr_px.second])
                continue;
            auto end_visited_check = hrs::now();
            vis_time += (end_visited_check - start_visited_check);

            if (mag_at_px(curr_px) > high_thresh) {
                std::queue<Px> to_visit({curr_px});
                visited[curr_px.first][curr_px.second] = true;

                auto start_while = hrs::now();
                while (!to_visit.empty()) {

                    auto start_mat_rw = hrs::now();

                    const auto px = to_visit.front();
                    to_visit.pop();
                    const auto curr_mag = mag_at_px(px);

                    if (curr_mag < low_thresh)
                        continue;

                    thresh_mag.at<std::uint8_t>(px.first - 1, px.second - 1) = curr_mag;

                    auto end_mat_rw = hrs::now();
                    mat_rw_time += (end_mat_rw - start_mat_rw);

                    auto start_neighbours = hrs::now();

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

                    auto end_neighbours = hrs::now();
                    neighbours_time += (end_neighbours - start_neighbours);
                }
                auto end_while = hrs::now();
                while_time += (end_while - start_while);
            }
        }
    }
    auto end_total  = hrs::now();
    auto total_time = end_total - start_total;

    auto to_perc = [](const Duration &child, const Duration &parent) {
        return static_cast<double>(std::chrono::duration_cast<Nanos>(child).count()) /
               std::chrono::duration_cast<Nanos>(parent).count() * 100;
    };

    std::println("Total: {}", total_time, total_time);
    std::println("\tVis check: {} ({:.2f}% of total)", vis_time, to_perc(vis_time, total_time));
    std::println("\t\tWhile: {} ({:.2f}% of total)", while_time, to_perc(while_time, total_time));
    std::println("\t\t\tMat ReadWr: {:>10} ({:.2f}% of while)", mat_rw_time, to_perc(mat_rw_time, while_time));
    std::println("\t\t\tNeighbours: {:>10} ({:.2f}% of while)", neighbours_time, to_perc(neighbours_time, while_time));

    return thresh_mag;
}
