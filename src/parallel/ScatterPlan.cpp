#include "ScatterPlan.h"

namespace tndm {

void ScatterPlan::setup(std::unordered_map<int, std::vector<std::size_t>> send_map,
                        std::unordered_map<int, std::vector<std::size_t>> recv_map) {
    const auto make_indices = [](auto const& map) {
        std::size_t size = 0;
        for (auto& [key, value] : map) {
            size += value.size();
        }

        auto indices = std::vector<std::size_t>{};
        indices.reserve(size);
        for (auto& [key, value] : map) {
            for (auto&& v : value) {
                indices.emplace_back(v);
            }
        }
        return indices;
    };

    send_indices_ = make_indices(send_map);
    recv_indices_ = make_indices(recv_map);

    const auto make_blocks = [](auto const& map) {
        auto blocks = std::vector<CommBlock>{};
        blocks.reserve(map.size());
        std::size_t offset = 0;
        for (auto& [key, value] : map) {
            int count = value.size();
            auto block = CommBlock{offset, count, key};
            blocks.emplace_back(CommBlock{offset, count, key});
            offset += count;
        }
        return blocks;
    };

    send_blocks_ = make_blocks(send_map);
    recv_blocks_ = make_blocks(recv_map);
}

} // namespace tndm
