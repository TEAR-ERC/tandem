#ifndef LOCALFACES_H
#define LOCALFACES_H

#include "MeshData.h"
#include "Simplex.h"
#include "util/Range.h"

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

    LocalFaces() : localSize_(0) {}
    LocalFaces(std::vector<Simplex<D>>&& faces, std::vector<int>&& owner,
               std::vector<std::size_t>&& contiguousGIDs, std::size_t localSize)
        : faces_(std::move(faces)), owner_(owner), l2cg_(std::move(contiguousGIDs)),
          localSize_(localSize) {
        assert(localSize_ <= faces_.size());
        makeG2LMap();
    }

    Simplex<D> const& operator[](std::size_t lid) const {
        assert(lid < size());
        return faces_[lid];
    }
    /**
     * @brief Number of faces in interior & ghost
     */
    std::size_t size() const { return faces_.size(); }
    /**
     * @brief Number of faces in interior
     *
     * Note that faces [0, localSize_) are interior and faces [localSize_, faces.size()) are ghost.
     */
    std::size_t localSize() const { return localSize_; }

    std::size_t owner(std::size_t lid) const {
        assert(lid < owner_.size());
        return owner_[lid];
    }
    std::size_t l2cg(std::size_t lid) const {
        assert(lid < l2cg_.size());
        return l2cg_[lid];
    }
    auto const& g2l() const { return g2l_; }
    auto const& contiguousGIDs() const { return l2cg_; }

    auto const& faces() const { return faces_; }
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

    void setRegionData(std::unique_ptr<MeshData> data) { regionData_ = std::move(data); }
    MeshData const* regionData() const { return regionData_.get(); }
	
	void permute(std::vector<std::size_t> const& permutation) {
        assert(permutation.size() == faces_.size());
        apply_permutation(faces_, permutation);
        apply_permutation(owner_, permutation);
        apply_permutation(l2cg_, permutation);
        if (meshData_) {
            meshData_->permute(permutation);
        }
        permuteSharedRanks(permutation);
        makeG2LMap();
    }

private:
    void permuteSharedRanks(std::vector<std::size_t> const& permutation) {
        // Permute shared ranks
        assert(permutation.size() == sharedRanksDispls_.size());
        std::vector<int> sharedRanksPermuted, sharedRanksCount;
        sharedRanksPermuted.reserve(sharedRanks_.size());
        sharedRanksCount.reserve(sharedRanksDispls_.size());
        for (auto&& p : permutation) {
            auto count = sharedRanksDispls_.count(p);
            sharedRanksCount.push_back(count);
            for (int i = sharedRanksDispls_[p], end = sharedRanksDispls_[p] + count; i < end; ++i) {
                sharedRanksPermuted.push_back(sharedRanks_[i]);
            }
        }
        sharedRanks_.swap(sharedRanksPermuted);
        sharedRanksDispls_ = Displacements(sharedRanksCount);
    }

    void makeG2LMap() {
        g2l_.clear();
        std::size_t local = 0;
        for (auto& f : faces_) {
            g2l_[f] = local++;
        }
    }

    std::vector<Simplex<D>> faces_;
    std::vector<int> owner_;
    l2cg_t l2cg_;
    std::size_t localSize_;
    g2l_t g2l_;
    std::vector<int> sharedRanks_;
    Displacements<int> sharedRanksDispls_;
    std::unique_ptr<MeshData> meshData_;
    std::unique_ptr<MeshData> regionData_;
};

} // namespace tndm

#endif // LOCALFACES_H
