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
    LocalFaces(std::vector<Simplex<D>>&& faces, Displacements<int>&& ranklayout)
        : faces(std::move(faces)), ranklayout(std::move(ranklayout)) {}

    std::vector<Simplex<D>> const& getFaces() const { return faces; }
    Displacements<int> const& getRankLayout() const { return ranklayout; }

    Simplex<D> const& operator[](std::size_t lid) const {
        assert(lid < size());
        return faces[lid];
    }
    std::size_t size() const { return faces.size(); }

    auto begin() const { return faces.cbegin(); }
    auto end() const { return faces.cend(); }

private:
    std::vector<Simplex<D>> faces;
    Displacements<int> ranklayout;
};

} // namespace tndm

#endif // LOCALFACES_H
