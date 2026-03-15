#include "doctest.h"

#include "form/DGCurvilinearCommon.h"

#include <array>
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
}
