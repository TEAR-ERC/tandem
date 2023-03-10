#ifndef GLOBALSIMPLEXMESHBUILDER_20200901_H
#define GLOBALSIMPLEXMESHBUILDER_20200901_H

#include "io/GMSHParser.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/Simplex.h"

#include <mpi.h>

#include <array>
#include <vector>

namespace tndm {

template <std::size_t D> struct GMSHSimplexType {};
template <> struct GMSHSimplexType<0u> { static constexpr std::array<long, 1> types = {15}; };
template <> struct GMSHSimplexType<1u> {
    static constexpr std::array<long, 10> types = {1, 8, 26, 27, 28, 62, 63, 64, 65, 66};
};
template <> struct GMSHSimplexType<2u> {
    static constexpr std::array<long, 10> types = {2, 9, 21, 23, 25, 42, 43, 44, 45, 46};
};
template <> struct GMSHSimplexType<3u> {
    static constexpr std::array<long, 10> types = {4, 11, 29, 30, 31, 71, 72, 73, 74, 75};
};

template <std::size_t D> bool is_gmsh_simplex(long type) {
    return std::find(GMSHSimplexType<D>::types.begin(), GMSHSimplexType<D>::types.end(), type) !=
           GMSHSimplexType<D>::types.end();
}

template <std::size_t D> struct is_lower_dimensional_gmsh_simplex {
    static bool value(long type) {
        return is_gmsh_simplex<D - 1u>(type) ||
               is_lower_dimensional_gmsh_simplex<D - 1u>::value(type);
    }
};
template <> struct is_lower_dimensional_gmsh_simplex<0> {
    static bool value(long type) { return false; }
};
template <std::size_t D> inline bool is_lower_dimensional_gmsh_simplex_v(long type) {
    return is_lower_dimensional_gmsh_simplex<D>::value(type);
}

template <std::size_t D> class GlobalSimplexMeshBuilder : public GMSHMeshBuilder {
private:
    constexpr static std::size_t NumVerts = D + 1u;

    std::vector<std::array<double, D>> vertices;
    std::vector<Simplex<D>> elements;
    std::vector<int> element_ptags;
    std::vector<int> element_etags;
    std::vector<Simplex<D - 1u>> facets;
    std::vector<int> facet_ptags;
    std::vector<int> facet_etags;
    Managed<Matrix<long>> high_order_nodes;
    Managed<Matrix<unsigned>> node_permutations_;

    std::size_t ignoredElems = 0;
    // std::size_t unknownBC = 0;
    std::size_t type_ = 0;

    void preparePermutationTable(std::size_t numNodes);

public:
    inline void setNumVertices(std::size_t numVertices) { vertices.resize(numVertices); }
    inline void setVertex(long id, std::array<double, 3> const& x) {
        for (std::size_t i = 0; i < D; ++i) {
            vertices[id][i] = x[i];
        }
    }
    inline void setNumElements(std::size_t numElements) {
        elements.reserve(numElements);
		element_ptags.reserve(numElements);
		element_etags.reserve(numElements);
        facets.reserve(numElements);
		facet_ptags.reserve(numElements);
		facet_etags.reserve(numElements);
        type_ = 0;
    }
    void addElement(long type, int ptag, int etag, long* node, std::size_t numNodes);

    // inline auto getUnknownBC() const { return unknownBC; }
    std::unique_ptr<GlobalSimplexMesh<D>> create(MPI_Comm comm);
};

} // namespace tndm

#endif // GLOBALSIMPLEXMESHBUILDER_20200901_H
