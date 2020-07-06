#ifndef LOCALFACES_H
#define LOCALFACES_H

#include "MeshData.h"
#include "Simplex.h"

#include "mneme/displacements.hpp"
#include "mneme/span.hpp"

#include <cassert>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

using mneme::Displacements;
using mneme::span;

namespace tndm {

template <std::size_t D> class LocalFaces {
public:
    using g2l_t = std::unordered_map<Simplex<D>, std::size_t, SimplexHash<D>>;
    using l2cg_t = std::vector<std::size_t>;

    LocalFaces() {}
    LocalFaces(std::vector<Simplex<D>>&& faces, std::vector<std::size_t>&& contiguousGIDs)
        : faces_(std::move(faces)), l2cg_(std::move(contiguousGIDs)) {
        makeG2LMap();
    }

    Simplex<D> const& operator[](std::size_t lid) const {
        assert(lid < size());
        return faces_[lid];
    }
    std::size_t size() const { return faces_.size(); }
    auto const& g2l() const { return g2l_; }
    auto const& l2cg(std::size_t lid) const {
        assert(lid < l2cg_.size());
        return l2cg_[lid];
    }

    auto begin() const { return faces_.cbegin(); }
    auto end() const { return faces_.cend(); }

    void setSharedRanks(std::vector<int>&& sharedRks, Displacements<int>&& sharedRksDispls) {
        sharedRanks_ = std::move(sharedRks);
        sharedRanksDispls_ = std::move(sharedRksDispls);
    }

    auto getSharedRanks(std::size_t lid) const {
        assert(lid < size());
        auto from = sharedRanksDispls_[lid];
        return span(&sharedRanks_[from], sharedRanksDispls_.count(lid));
    }

    void setMeshData(std::unique_ptr<MeshData> data) { meshData_ = std::move(data); }
    MeshData const* data() const { return meshData_.get(); }

private:
    void makeG2LMap() {
        g2l_.clear();
        std::size_t local = 0;
        for (auto& f : faces_) {
            g2l_[f] = local++;
        }
    }

    std::vector<Simplex<D>> faces_;
    l2cg_t l2cg_;
    g2l_t g2l_;
    std::vector<int> sharedRanks_;
    Displacements<int> sharedRanksDispls_;
    std::unique_ptr<MeshData> meshData_;
};

} // namespace tndm

#endif // LOCALFACES_H
