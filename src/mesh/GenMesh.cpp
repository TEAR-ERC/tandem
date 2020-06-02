#include "GenMesh.h"
#include <utility>

namespace tndm {

template <>
std::array<GenMesh<1>::simplex_t, 1> GenMesh<1>::tessellate(std::array<uint64_t, 2> const& vertGIDs,
                                                            bool) {
    std::array<simplex_t, 1> result;
    result[0] = vertGIDs;
    return result;
}

template <>
std::array<GenMesh<2>::simplex_t, 2> GenMesh<2>::tessellate(std::array<uint64_t, 4> const& vertGIDs,
                                                            bool isOdd) {
    std::array<simplex_t, 2> result;
    result[0] = {vertGIDs[0], vertGIDs[1], vertGIDs[2]};
    result[1] = {vertGIDs[1], vertGIDs[3], vertGIDs[2]};
    return result;
}

template <>
std::array<GenMesh<3>::simplex_t, 5> GenMesh<3>::tessellate(std::array<uint64_t, 8> const& vertGIDs,
                                                            bool isOdd) {
    std::array<simplex_t, 5> result;
    if (isOdd) {
        result[0] = {vertGIDs[0], vertGIDs[1], vertGIDs[2], vertGIDs[4]};
        result[1] = {vertGIDs[1], vertGIDs[2], vertGIDs[3], vertGIDs[7]};
        result[2] = {vertGIDs[2], vertGIDs[4], vertGIDs[6], vertGIDs[7]};
        result[3] = {vertGIDs[1], vertGIDs[4], vertGIDs[5], vertGIDs[7]};
        result[4] = {vertGIDs[2], vertGIDs[4], vertGIDs[1], vertGIDs[7]};
    } else {
        result[0] = {vertGIDs[0], vertGIDs[2], vertGIDs[3], vertGIDs[6]};
        result[1] = {vertGIDs[0], vertGIDs[1], vertGIDs[3], vertGIDs[5]};
        result[2] = {vertGIDs[0], vertGIDs[4], vertGIDs[5], vertGIDs[6]};
        result[3] = {vertGIDs[3], vertGIDs[5], vertGIDs[6], vertGIDs[7]};
        result[4] = {vertGIDs[0], vertGIDs[3], vertGIDs[5], vertGIDs[6]};
    }
    return result;
}

template <std::size_t D> std::unique_ptr<typename GenMesh<D>::mesh_t> GenMesh<D>::uniformMesh() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::size_t numVertsGlobal = 1u;
    for (auto& np1 : Np1) {
        numVertsGlobal *= np1;
    }

    // A vertex has multi-index (v_1,...,v_d)
    // For parallel mesh generation we flatten the index and equally distribute vertices
    auto vertsLocal = distribute(Range(numVertsGlobal), rank, size);

    const std::size_t numVertices = vertsLocal.length();
    std::vector<vertex_t> vertices(numVertices);

    // Map global vertex id to position
    auto vertex_pos = [&](std::array<uint64_t, D> const& v) {
        vertex_t vert;
        for (std::size_t d = 0; d < D; ++d) {
            vert[d] = static_cast<double>(v[d]) / N[d];
        }
        return vert;
    };

    for (auto vflat : vertsLocal) {
        auto v = unflatten(vflat, Np1);
        vertices[vflat - vertsLocal.from] = vertex_pos(v);
    }

    // elements live on grid with size N_1 x ... x N_d
    unsigned numElemsGlobal = 1u;
    for (auto& n : N) {
        numElemsGlobal *= n;
    }

    // An element (cuboid for D=3) has multi-index (n_1,...,n_d)
    // For parallel mesh generation we flatten the index and equally distribute elements
    auto elemsLocal = distribute(Range(numElemsGlobal), rank, size);

    // We have 2 vertices per dim i.e. 2^D vertices per element
    constexpr uint64_t numVertGIDs = (1 << D);
    std::array<uint64_t, D> D2;
    std::fill(D2.begin(), D2.end(), 2);

    const std::size_t numElements = elemsLocal.length() * TessInfo<D>::NumSimplices;
    std::vector<simplex_t> elements(numElements);

    for (auto eflat : elemsLocal) {
        auto e = unflatten(eflat, N);
        std::array<uint64_t, numVertGIDs> vertGIDs;
        // Get 2^D vertices
        for (uint64_t vflat = 0; vflat < numVertGIDs; ++vflat) {
            auto v = e + unflatten(vflat, D2);
            vertGIDs[vflat] = flatten(v, Np1);
        }
        bool isOdd = false;
        for (auto&& ee : e) {
            isOdd ^= (ee % 2 == 1);
        }

        // Tessellate cuboid into D-simplices
        auto simplices = tessellate(vertGIDs, isOdd);
        std::copy(simplices.begin(), simplices.end(),
                  elements.begin() + TessInfo<D>::NumSimplices * (eflat - elemsLocal.from));
    }

    auto vertexData = std::make_unique<vertex_data_t>(std::move(vertices));
    auto mesh = std::make_unique<mesh_t>(std::move(elements), std::move(vertexData));
    auto boundaryMesh = extractBoundaryMesh(*mesh);
    mesh->template setBoundaryMesh<D - 1>(std::move(boundaryMesh));
    return mesh;
}

template <std::size_t D>
std::unique_ptr<typename GenMesh<D>::boundary_mesh_t>
GenMesh<D>::extractBoundaryMesh(mesh_t const& mesh) {
    std::vector<boundary_simplex_t> boundaryElements;
    std::vector<int> boundaryConditions;
    for (auto& elem : mesh.getElements()) {
        for (auto& face : elem.downward()) {
            for (std::size_t d = 0; d < D; ++d) {
                bool faceAtBnd = true;
                for (auto& vflat : face) {
                    auto v = unflatten(vflat, Np1);
                    faceAtBnd = faceAtBnd && ((v[d] == 0) || (v[d] == Np1[d] - 1));
                }
                if (faceAtBnd) {
                    boundaryElements.emplace_back(face);
                    boundaryConditions.emplace_back(d + 1);
                }
            }
        }
    }
    auto boundaryData = std::make_unique<boundary_data_t>(std::move(boundaryConditions));
    return std::make_unique<boundary_mesh_t>(std::move(boundaryElements), nullptr,
                                             std::move(boundaryData));
}

template class GenMesh<2u>;
template class GenMesh<3u>;

} // namespace tndm
