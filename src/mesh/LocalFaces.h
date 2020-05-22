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
    LocalFaces() {}
    LocalFaces(std::vector<Simplex<D>>&& faces) : faces(std::move(faces)) {}

    Simplex<D> const& operator[](std::size_t lid) const {
        assert(lid < size());
        return faces[lid];
    }
    std::size_t size() const { return faces.size(); }

    auto begin() const { return faces.cbegin(); }
    auto end() const { return faces.cend(); }

    void setSharedRanks(std::vector<int>&& sharedRks, Displacements<int>&& sharedRksDispls) {
        sharedRanks = std::move(sharedRks);
        sharedRanksDispls = std::move(sharedRksDispls);
    }

    auto getSharedRanks(std::size_t lid) const {
        assert(lid < size());
        auto from = sharedRanksDispls[lid];
        return span(&sharedRanks[from], sharedRanksDispls.count(lid));
    }

    void setMeshData(std::unique_ptr<MeshData> data) { meshData = std::move(data); }
    MeshData const* data() const { return meshData.get(); }

private:
    std::vector<Simplex<D>> faces;
    std::vector<int> sharedRanks;
    Displacements<int> sharedRanksDispls;
    std::unique_ptr<MeshData> meshData;
};

} // namespace tndm

#endif // LOCALFACES_H
