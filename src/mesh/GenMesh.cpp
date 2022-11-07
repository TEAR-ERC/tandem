#include "GenMesh.h"
#include "parallel/Distribute.h"
#include "util/MultiIndex.h"
#include "util/Range.h"

#include <algorithm>
#include <iterator>
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
    if (isOdd) {
        result[0] = {vertGIDs[0b00], vertGIDs[0b01], vertGIDs[0b11]};
        result[1] = {vertGIDs[0b00], vertGIDs[0b10], vertGIDs[0b11]};
    } else {
        result[0] = {vertGIDs[0b00], vertGIDs[0b01], vertGIDs[0b10]};
        result[1] = {vertGIDs[0b01], vertGIDs[0b10], vertGIDs[0b11]};
    }
    return result;
}

template <>
std::array<GenMesh<3>::simplex_t, 5> GenMesh<3>::tessellate(std::array<uint64_t, 8> const& vertGIDs,
                                                            bool isOdd) {
    std::array<simplex_t, 5> result;
    if (isOdd) {
        result[0] = {vertGIDs[0b000], vertGIDs[0b001], vertGIDs[0b010], vertGIDs[0b100]};
        result[1] = {vertGIDs[0b001], vertGIDs[0b010], vertGIDs[0b011], vertGIDs[0b111]};
        result[2] = {vertGIDs[0b010], vertGIDs[0b100], vertGIDs[0b110], vertGIDs[0b111]};
        result[3] = {vertGIDs[0b001], vertGIDs[0b100], vertGIDs[0b101], vertGIDs[0b111]};
        result[4] = {vertGIDs[0b010], vertGIDs[0b100], vertGIDs[0b001], vertGIDs[0b111]};
    } else {
        result[0] = {vertGIDs[0b000], vertGIDs[0b010], vertGIDs[0b011], vertGIDs[0b110]};
        result[1] = {vertGIDs[0b000], vertGIDs[0b001], vertGIDs[0b011], vertGIDs[0b101]};
        result[2] = {vertGIDs[0b000], vertGIDs[0b100], vertGIDs[0b101], vertGIDs[0b110]};
        result[3] = {vertGIDs[0b011], vertGIDs[0b101], vertGIDs[0b110], vertGIDs[0b111]};
        result[4] = {vertGIDs[0b000], vertGIDs[0b011], vertGIDs[0b101], vertGIDs[0b110]};
    }
    return result;
}

template <std::size_t D>
std::unique_ptr<typename GenMesh<D>::mesh_t> GenMesh<D>::uniformMesh() const {
    int rank, size;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);

    uint64_t numVertsGlobal = 1u;
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
            std::size_t region = findRegion(d, v);
            double h = (points_[d][region + 1] - points_[d][region]) /
                       (regions_[d][region + 1] - regions_[d][region]);
            vert[d] = points_[d][region] + static_cast<double>(v[d] - regions_[d][region]) * h;
        }
        return vert;
    };

    for (auto vflat : vertsLocal) {
        auto v = unflatten(vflat, Np1);
        vertices[vflat - vertsLocal.from] = vertex_pos(v);
    }

    // elements live on grid with size N_1 x ... x N_d
    uint64_t numElemsGlobal = 1u;
    for (auto& n : N) {
        numElemsGlobal *= n;
    }
    numElemsGlobal *= TessInfo<D>::NumSimplices;

    // An element (cuboid for D=3) has multi-index (n_1,...,n_d)
    // For parallel mesh generation we flatten the index and equally distribute elements
    auto elemsLocal = distribute(Range(numElemsGlobal), rank, size);

    // We have 2 vertices per dim i.e. 2^D vertices per element
    constexpr uint64_t numVertGIDs = (1 << D);
    std::array<uint64_t, D> D2;
    std::fill(D2.begin(), D2.end(), 2);

    const std::size_t numElements = elemsLocal.length();
    std::vector<simplex_t> elements(numElements);
	
    uint64_t ecubeFrom = elemsLocal.from / TessInfo<D>::NumSimplices;
    uint64_t ecubeTo = 1 + (static_cast<int64_t>(elemsLocal.to) - 1) / TessInfo<D>::NumSimplices;
    for (uint64_t ecube = ecubeFrom; ecube < ecubeTo; ++ecube) {
        auto e = unflatten(ecube, N);
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
        int64_t firstPlex =
            ecube == ecubeFrom ? elemsLocal.from - ecubeFrom * TessInfo<D>::NumSimplices : 0;
        int64_t lastPlex =
            ecube == ecubeTo - 1 ? ecubeTo * TessInfo<D>::NumSimplices - elemsLocal.to : 0;
        std::copy(simplices.begin() + firstPlex,
                  simplices.begin() + (TessInfo<D>::NumSimplices - lastPlex),
                  elements.begin() + ecube * TessInfo<D>::NumSimplices + firstPlex -
                      elemsLocal.from);
    }
	
	std::vector<int> regions(elements.length(), 0); // default region = 0

    auto vertexData = std::make_unique<vertex_data_t>(std::move(vertices));
	auto regionData = std::make_unique<region_data_t>(std::move(regions));
    auto mesh =
        std::make_unique<mesh_t>(std::move(elements), std::move(vertexData), nullptr, std::move(regionData), comm_);
    auto boundaryMesh = extractBoundaryMesh(*mesh);
    mesh->template setBoundaryMesh<D - 1>(std::move(boundaryMesh));
    return mesh;
}

template <std::size_t D>
std::unique_ptr<typename GenMesh<D>::boundary_mesh_t>
GenMesh<D>::extractBoundaryMesh(mesh_t const& mesh) const {
    std::vector<boundary_simplex_t> boundaryElements;
    std::vector<BC> boundaryConditions;
    for (auto& elem : mesh.getElements()) {
        for (auto& face : elem.downward()) {
            auto v0 = unflatten(face[0], Np1);
            std::array<bool, D> inPlane;
            inPlane.fill(true);
            for (auto vflat = face.begin() + 1; vflat != face.end(); ++vflat) {
                auto v = unflatten(*vflat, Np1);
                for (std::size_t d = 0; d < D; ++d) {
                    inPlane[d] = inPlane[d] && (v[d] == v0[d]);
                }
            }
            assert(std::count(inPlane.begin(), inPlane.end(), true) <= 1);
            auto it = std::find(inPlane.begin(), inPlane.end(), true);
            if (it != inPlane.end()) {
                auto dim = std::distance(inPlane.begin(), it);
                auto planeIt = std::find(regions_[dim].begin(), regions_[dim].end(), v0[dim]);
                if (planeIt != regions_[dim].end()) {
                    auto plane = std::distance(regions_[dim].begin(), planeIt);
                    boundaryElements.emplace_back(face);
                    std::array<std::size_t, D - 1u> otherRegions{};
                    for (auto&& vflat : face) {
                        auto v = unflatten(vflat, Np1);
                        auto it = otherRegions.begin();
                        for (std::size_t d = 0; d < D; ++d) {
                            if (dim != d) {
                                *it = std::max(*it, findRegion(d, v));
                                ++it;
                            }
                        }
                    }
                    boundaryConditions.emplace_back(bcs_[dim](plane, otherRegions));
                }
            }
        }
    }
    auto boundaryData = std::make_unique<boundary_data_t>(std::move(boundaryConditions));
    return std::make_unique<boundary_mesh_t>(std::move(boundaryElements), nullptr,
                                             std::move(boundaryData), nullptr, comm_);
}

template class GenMesh<2u>;
template class GenMesh<3u>;

} // namespace tndm
