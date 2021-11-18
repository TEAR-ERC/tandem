#include "Equidistant.h"
#include "geometry/Vector.h"
#include "mesh/Simplex.h"
#include "util/Combinatorics.h"
#include <cassert>

namespace tndm {

template <std::size_t D>
template <std::size_t DD>
std::array<std::array<double, D>, DD + 1> EquidistantNodesFactory<D>::shrinkAndShift(
    double factor, std::array<std::array<double, D>, DD + 1> const& verts) const {
    // New barycentric coordinates for result vertex i are
    //               | 1.0 - DD * factor  if  i == j
    // lambda_{ij} = |
    //               | factor             if  i != j
    std::array<std::array<double, D>, DD + 1> result{};
    for (unsigned i = 0; i <= DD; ++i) {
        for (unsigned j = 0; j <= DD; ++j) {
            double lambda_ij = (i == j) ? 1.0 - DD * factor : factor;
            result[i] = result[i] + lambda_ij * verts[j];
        }
    }
    return result;
}

template <std::size_t D>
void EquidistantNodesFactory<D>::edge(int n, std::array<std::array<double, D>, 2> const& verts,
                                      std::vector<std::array<double, D>>& result) const {
    double h = 1.0 / (n + 1);
    for (int nn = 1; nn <= n; ++nn) {
        double L = nn * h;
        result.push_back((1.0 - L) * verts[0] + L * verts[1]);
    }
}

template <std::size_t D>
void EquidistantNodesFactory<D>::triangle(int n, std::array<std::array<double, D>, 3>&& verts,
                                          std::vector<std::array<double, D>>& result) const {
    while (n >= 0) {
        result.push_back(verts[0]);
        if (n > 0) {
            result.push_back(verts[1]);
            result.push_back(verts[2]);

            if (n >= 1) {
                edge(n - 1, {verts[0], verts[1]}, result);
                edge(n - 1, {verts[1], verts[2]}, result);
                edge(n - 1, {verts[2], verts[0]}, result);
            }

            verts = shrinkAndShift<2>(1.0 / n, verts);
        }
        n -= 3;
    }
}

template <std::size_t D>
void EquidistantNodesFactory<D>::tet(int n, std::array<std::array<double, D>, 4>&& verts,
                                     std::vector<std::array<double, D>>& result) const {
    while (n >= 0) {
        result.push_back(verts[0]);
        if (n > 0) {
            result.push_back(verts[1]);
            result.push_back(verts[2]);
            result.push_back(verts[3]);

            if (n >= 1) {
                for (auto const& e : tet_edge_convention(convention_)) {
                    edge(n - 1, {verts[e[0]], verts[e[1]]}, result);
                }
            }

            double sf = 1.0 / n;
            if (n >= 3) {
                for (auto const& f : tet_face_convention(convention_)) {
                    triangle(n - 3, shrinkAndShift<2>(sf, {verts[f[0]], verts[f[1]], verts[f[2]]}),
                             result);
                }
            }

            verts = shrinkAndShift<3>(sf, verts);
        }
        n -= 4;
    }
}

template <std::size_t D>
std::vector<std::array<double, D>> EquidistantNodesFactory<D>::operator()(unsigned degree) const {
    unsigned numNodes = binom(degree + D, D);
    std::vector<std::array<double, D>> result;
    result.reserve(numNodes);

    using vtx_t = std::array<double, D>;
    if constexpr (D == 1) {
        auto refVertices = Simplex<D>::referenceSimplexVertices();
        result.push_back(refVertices[0]);
        result.push_back(refVertices[1]);
        edge(degree - 1, refVertices, result);
    } else if constexpr (D == 2) {
        triangle(degree, Simplex<D>::referenceSimplexVertices(), result);
    } else if constexpr (D == 3) {
        tet(degree, Simplex<D>::referenceSimplexVertices(), result);
    }
    assert(numNodes == result.size());
    return result;
}

template class EquidistantNodesFactory<1u>;
template class EquidistantNodesFactory<2u>;
template class EquidistantNodesFactory<3u>;

} // namespace tndm
