#ifndef LOCALSIMPLEXMESH_H
#define LOCALSIMPLEXMESH_H

#include <cstddef>
#include <vector>
#include <utility>

#include <Eigen/Core>

#include "Simplex.h"

namespace tndm {

template<std::size_t D>
class LocalSimplexMesh {
public:
    using vertex_t = Eigen::Matrix<double, 3, 1>;
    using simplex_t = Simplex<D>;

    LocalSimplexMesh(std::vector<vertex_t>&& vertices, std::vector<Simplex<D>>&& elements)
        : verts(std::move(vertices)), elems(std::move(elements)) {}

    template<unsigned d>
    std::array<vertex_t,d+1> vertices(unsigned lid) const {
        static_assert(d == D);

        auto& plex = elems[lid];
        std::array<vertex_t,d+1> vs;
        auto out = vs.begin();
        for (auto& p : plex) {
            (*out)++ = verts[p];
        }
        return vs;
    }

    template<typename RealT, std::size_t Dout = D>
    std::vector<RealT> flatVertices() const {
        static_assert(Dout >= D);
        std::vector<RealT> vout;
        vout.reserve(Dout*verts.size());
        for (auto& v : verts) {
            std::size_t d;
            for (d = 0; d < D; ++d) {
                vout.push_back(static_cast<RealT>(v[d]));
            }
            for (; d < Dout; ++d) {
                vout.push_back(0.0);
            }
        }
        return vout;
    }

    template<typename IntT>
    std::vector<IntT> flatElements() const {
        std::vector<IntT> eout;
        eout.reserve((D+1)*elems.size());
        for (auto& e : elems) {
            for (auto& p : e) {
                eout.push_back(static_cast<IntT>(p));
            }
        }
        return eout;
    }

    std::size_t numVertices() const {
        return verts.size();
    }

    std::size_t numElements() const {
        return elems.size();
    }

private:
    std::vector<vertex_t> verts;
    std::vector<Simplex<D>> elems;
};

}

#endif // LOCALSIMPLEXMESH_H
