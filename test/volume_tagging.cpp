#include "doctest.h"

#include "form/DGCurvilinearCommon.h"
#include "geometry/Curvilinear.h"
#include "io/GlobalSimplexMeshBuilder.h"
#include "mesh/GenMesh.h"
#include "mesh/MeshData.h"
#include "parallel/CommPattern.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <array>
#include <limits>
#include <memory>
#include <mpi.h>
#include <vector>

using namespace tndm;

TEST_CASE("Volume tagging") {
    SUBCASE("VolumeData round-trip") {
        std::vector<long int> tags = {3, 1, 5};
        VolumeData vd(std::move(tags));

        std::vector<std::size_t> lids = {0, 1, 2};
        // single-rank test - one entry in sendcounts
        std::vector<int> sendcounts(1, static_cast<int>(lids.size()));
        AllToAllV a2a(std::move(sendcounts), MPI_COMM_WORLD);

        auto newMeshData = vd.redistributed(lids, a2a);
        auto const& outTags = dynamic_cast<VolumeData const&>(*newMeshData).getVolumeTags();
        CHECK(outTags.size() == 3);
        CHECK(outTags[0] == 3);
        CHECK(outTags[1] == 1);
        CHECK(outTags[2] == 5);
    }
    SUBCASE("VolumeData sends -1 for out-of-range lids") {
        std::vector<long int> tags = {7};
        VolumeData vd(std::move(tags));

        std::vector<std::size_t> lids = {std::numeric_limits<std::size_t>::max()};
        std::vector<int> sendcounts(1, static_cast<int>(lids.size()));
        AllToAllV a2a(std::move(sendcounts), MPI_COMM_WORLD);

        auto newMeshData = vd.redistributed(lids, a2a);
        auto const& outTags = dynamic_cast<VolumeData const&>(*newMeshData).getVolumeTags();
        REQUIRE(outTags.size() == 1);
        CHECK(outTags[0] == -1);
    }
    SUBCASE("make_volume_functional switches on volume tag") {
        constexpr std::size_t D = 2u;

        // 2-triangle unit square, tags 1 and 2
        GlobalSimplexMeshBuilder<D> builder;
        builder.setNumVertices(4);
        builder.setVertex(0, {0.0, 0.0, 0.0});
        builder.setVertex(1, {1.0, 0.0, 0.0});
        builder.setVertex(2, {1.0, 1.0, 0.0});
        builder.setVertex(3, {0.0, 1.0, 0.0});

        builder.setNumElements(2);
        long nodes0[] = {0, 1, 2};
        long nodes1[] = {0, 2, 3};
        long nodes2[] = {0, 1};
        long nodes3[] = {1, 2}; // interface between two elements
        long nodes4[] = {2, 3};
        long nodes5[] = {3, 0};
        builder.addElement(2, 1, nodes0, 3); // tag=1
        builder.addElement(2, 2, nodes1, 3); // tag=2
        builder.addElement(1, static_cast<long>(BC::None), nodes2, 2);
        builder.addElement(1, static_cast<long>(BC::None), nodes3, 2);
        builder.addElement(1, static_cast<long>(BC::None), nodes4, 2);
        builder.addElement(1, static_cast<long>(BC::None), nodes5, 2);

        auto globalMesh = builder.create(MPI_COMM_WORLD);
        auto mesh = globalMesh->getLocalMesh();

        auto transform = [](typename Curvilinear<D>::vertex_t const& v) { return v; };
        auto clptr = std::make_shared<Curvilinear<D>>(*mesh, transform);

        DGCurvilinearCommon<D> dg(clptr, 1u);

        std::size_t numElements = mesh->numElements();
        std::size_t numLocalElements = mesh->elements().localSize();
        std::size_t numLocalFacets = mesh->facets().localSize();
        dg.begin_preparation(numElements, numLocalElements, numLocalFacets);

        alignas(64) double mem[4096];
        LinearAllocator<double> scratch(mem, mem + 4096, 64);
        for (std::size_t el = 0; el < numElements; ++el) {
            scratch.reset();
            dg.prepare_volume(el, scratch);
        }

        auto fun = [](std::array<double, D> const&, long int tag) {
            std::array<double, 1> out{};
            out[0] = (tag == 1 ? 2.0 : 5.0);
            return out;
        };

        auto vf = dg.make_volume_functional<1>(fun);
        std::size_t Q = dg.volQuadratureRule().size();
        Managed<Matrix<double>> F(1, Q);

        // element 0 -> tag 1 -> 2.0
        vf(0, F);
        for (std::size_t q = 0; q < Q; ++q) {
            CHECK(F(0, q) == doctest::Approx(2.0));
        }

        // element 1 -> tag 2 -> 5.0
        vf(1, F);
        for (std::size_t q = 0; q < Q; ++q) {
            CHECK(F(0, q) == doctest::Approx(5.0));
        }
    }
}
