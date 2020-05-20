#ifndef LOCALFACES_H
#define LOCALFACES_H

#include "Simplex.h"

#include "mneme/displacements.hpp"
#include "mneme/span.hpp"

#include <cassert>
#include <cstddef>
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

    void setSharedRanks(std::vector<int>&& sharedRks, Displacements<int>&& sharedRksDispls) {
        sharedRanks = std::move(sharedRks);
        sharedRanksDispls = std::move(sharedRksDispls);
    }

    auto getSharedRanks(std::size_t lid) const {
        assert(lid < size());
        auto from = sharedRanksDispls[lid];
        return span(&sharedRanks[from], sharedRanksDispls.count(lid));
    }

    void setG2L(std::unordered_map<int, int> const* G2L) { g2l = G2L; }

    Simplex<D> const& operator[](std::size_t lid) const {
        assert(lid < size());
        return faces[lid];
    }
    std::size_t size() const { return faces.size(); }

    auto begin() const { return faces.cbegin(); }
    auto end() const { return faces.cend(); }

private:
    std::vector<Simplex<D>> faces;
    std::unordered_map<int, int> const* g2l = nullptr;
    std::vector<int> sharedRanks;
    Displacements<int> sharedRanksDispls;
};

} // namespace tndm

#endif // LOCALFACES_H
