#ifndef WRITER_20200618_H
#define WRITER_20200618_H

#include "mesh/LocalSimplexMesh.h"

#include <cstddef>
#include <vector>

template <typename IntT, std::size_t D>
std::vector<IntT> flatElements(tndm::LocalSimplexMesh<D> const& mesh) {
    std::vector<IntT> eout;
    eout.reserve((D + 1) * mesh.elements().size());
    for (auto& elem : mesh.elements()) {
        auto lids = mesh.template downward<0>(elem);
        for (auto& lid : lids) {
            eout.push_back(static_cast<IntT>(lid));
        }
    }
    return eout;
}
template <typename RealT, std::size_t D, std::size_t Dout, typename Func>
std::vector<RealT> flatVertices(std::vector<std::array<double, D>> const& verts, Func transform) {
    static_assert(Dout >= D);
    std::vector<RealT> vout;
    vout.reserve(Dout * verts.size());
    for (auto& v : verts) {
        std::size_t d;
        auto vt = transform(v);
        for (d = 0; d < D; ++d) {
            vout.push_back(static_cast<RealT>(vt[d]));
        }
        for (; d < Dout; ++d) {
            vout.push_back(0.0);
        }
    }
    return vout;
}

#endif // WRITER_20200618_H
