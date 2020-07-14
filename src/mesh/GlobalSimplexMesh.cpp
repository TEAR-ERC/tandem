#include "GlobalSimplexMesh.h"
#include "parallel/SortedDistribution.h"

namespace tndm {

template <std::size_t D> void GlobalSimplexMesh<D>::repartition() {
    auto distCSR = distributedCSR<idx_t>();
    auto partition = MetisPartitioner::partition(distCSR, D, 1.05, comm);

    doPartition(partition);
    isPartitionedByHash = false;
}

template <std::size_t D> void GlobalSimplexMesh<D>::repartitionByHash() {
    if (isPartitionedByHash) {
        return;
    }
    std::vector<idx_t> partition;
    partition.reserve(numElements());
    auto plex2rank = getPlex2Rank<D>();
    for (auto& e : elems_) {
        partition.emplace_back(plex2rank(e));
    }

    doPartition(partition);
    isPartitionedByHash = true;
}

template <std::size_t D>
void GlobalSimplexMesh<D>::doPartition(std::vector<idx_t> const& partition) {
    int procs, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);
    assert(partition.size() == numElements());

    std::vector<std::size_t> enumeration(partition.size());
    std::iota(enumeration.begin(), enumeration.end(), std::size_t(0));

    std::sort(enumeration.begin(), enumeration.end(),
              [&partition](std::size_t a, std::size_t b) { return partition[a] < partition[b]; });

    std::vector<int> sendcounts(procs, 0);
    std::vector<simplex_t> elemsToSend;
    elemsToSend.reserve(numElements());
    auto eIt = enumeration.begin();
    for (int p = 0; p < procs; ++p) {
        while (eIt != enumeration.end() && partition[*eIt] <= p) {
            ++sendcounts[p];
            elemsToSend.emplace_back(elems_[*eIt]);
            ++eIt;
        }
    }
    assert(eIt == enumeration.end());

    AllToAllV a2a(std::move(sendcounts), comm);
    mpi_array_type<uint64_t> mpi_simplex_t(D + 1);
    elems_ = a2a.exchange(elemsToSend, mpi_simplex_t.get());

    if (elementData) {
        elementData = elementData->redistributed(enumeration, a2a);
    }
}

template <std::size_t D>
LocalFaces<D>
GlobalSimplexMesh<D>::getGhostElements(std::vector<Simplex<D>> elems, unsigned overlap,
                                       std::vector<std::size_t> const& elemDist) const {
    int rank, procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);
    auto const myComm = comm;

    std::vector<std::size_t> cGIDs(elems.size());
    std::iota(cGIDs.begin(), cGIDs.end(), elemDist[rank]);

    if (overlap == 0) {
        auto lf = LocalFaces<D>(std::move(elems), std::move(cGIDs), elems.size());
        setSharedRanksAndElementData(lf, elemDist);
        return lf;
    }

    mpi_array_type<uint64_t> mpi_facet_t(D);
    mpi_array_type<uint64_t> mpi_elem_t(D + 1u);
    auto plex2rank = getPlex2Rank<D - 1u>();

    auto const makeDistributedUpwardMap = [&procs, &myComm, &mpi_facet_t, &mpi_elem_t, &plex2rank](
                                              auto&& elems, std::vector<std::size_t> const& cGIDs) {
        std::unordered_multimap<Simplex<D - 1u>, std::size_t, SimplexHash<D - 1u>> up;
        for (std::size_t elNo = 0; elNo < elems.size(); ++elNo) {
            auto downward = elems[elNo].template downward<D - 1u>();
            for (auto& s : downward) {
                up.emplace(s, elNo);
            }
        }
        std::vector<int> counts(procs, 0);
        for (auto&& u : up) {
            ++counts[plex2rank(u.first)];
        }
        std::vector<std::ptrdiff_t> offsets(procs + 1);
        offsets[0] = 0;
        std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);

        std::vector<Simplex<D - 1u>> upFaces(offsets.back());
        std::vector<Simplex<D>> upElems(offsets.back());
        std::vector<std::size_t> upCGIDs(offsets.back());
        for (auto&& u : up) {
            auto rank = plex2rank(u.first);
            upFaces[offsets[rank]] = u.first;
            upElems[offsets[rank]] = elems[u.second];
            upCGIDs[offsets[rank]] = cGIDs[u.second];
            ++offsets[rank];
        }

        AllToAllV a2a(std::move(counts), myComm);
        auto receivedUpFaces = a2a.exchange(upFaces, mpi_facet_t.get());
        auto receivedUpElems = a2a.exchange(upElems, mpi_elem_t.get());
        auto receivedUpCGIDs = a2a.exchange(upCGIDs);

        std::unordered_multimap<Simplex<D - 1u>, std::pair<std::size_t, Simplex<D>>,
                                SimplexHash<D - 1u>>
            distUp;
        for (std::size_t i = 0; i < receivedUpFaces.size(); ++i) {
            distUp.emplace(receivedUpFaces[i],
                           std::make_pair(receivedUpCGIDs[i], receivedUpElems[i]));
        }
        return distUp;
    };

    auto distUp = makeDistributedUpwardMap(elems, cGIDs);

    std::size_t numLocal = elems.size();

    for (int level = 0; level < static_cast<int>(overlap); ++level) {
        auto up = getBoundaryFaces(elems);

        auto const getRequestedFacesAndMissingVertices = [&procs, &myComm, &mpi_facet_t,
                                                          &plex2rank](auto&& elems, auto&& up) {
            std::vector<int> counts(procs, 0);
            for (auto&& u : up) {
                ++counts[plex2rank(u.first)];
            }
            std::vector<std::ptrdiff_t> offsets(procs + 1);
            offsets[0] = 0;
            std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);

            std::vector<Simplex<D - 1u>> boundaryFaces(offsets.back());
            std::vector<uint64_t> missingVertex(offsets.back());
            for (auto&& u : up) {
                // Find missing vertex i.e. elem \ facet
                auto& elem = elems[u.second];
                auto f = u.first.begin();
                auto e = elem.begin();
                while (e != elem.end() && f != u.first.end() && *e == *f) {
                    ++f;
                    ++e;
                }
                assert(e != elem.end());

                auto rank = plex2rank(u.first);
                boundaryFaces[offsets[rank]] = u.first;
                missingVertex[offsets[rank]] = *e;
                ++offsets[rank];
            }

            AllToAllV a2a(std::move(counts), myComm);
            auto requestedBoundaryFaces = a2a.exchange(boundaryFaces, mpi_facet_t.get());
            auto requestedMissingVertices = a2a.exchange(missingVertex);
            return std::make_tuple(requestedBoundaryFaces, requestedMissingVertices, a2a);
        };

        auto [requestedBoundaryFaces, requestedMissingVertices, a2a] =
            getRequestedFacesAndMissingVertices(elems, up);

        std::vector<Simplex<D>> requestedBoundaryElems(requestedBoundaryFaces.size());
        std::vector<std::size_t> requestedCGIDs(requestedBoundaryFaces.size());
        for (std::size_t i = 0; i < requestedBoundaryFaces.size(); ++i) {
            auto [it, end] = distUp.equal_range(requestedBoundaryFaces[i]);
            while (it != end && std::find(it->second.second.begin(), it->second.second.end(),
                                          requestedMissingVertices[i]) != it->second.second.end()) {
                ++it;
            }
            // Neighbour could not be found (i.e. domain boundary)
            if (it == end) {
                requestedCGIDs[i] = std::numeric_limits<std::size_t>::max();
                requestedBoundaryElems[i] = Simplex<D>::invalidSimplex();
            } else {
                requestedCGIDs[i] = it->second.first;
                requestedBoundaryElems[i] = it->second.second;
            }
        }
        a2a.swap();
        auto boundaryElems = a2a.exchange(requestedBoundaryElems, mpi_elem_t.get());
        auto boundaryCGIDs = a2a.exchange(requestedCGIDs);

        // Remove duplicates
        std::vector<std::size_t> enumeration(boundaryElems.size());
        std::iota(enumeration.begin(), enumeration.end(), std::size_t(0));
        std::sort(enumeration.begin(), enumeration.end(),
                  [&boundaryElems](std::size_t a, std::size_t b) {
                      return boundaryElems[a] < boundaryElems[b];
                  });
        apply_permutation(boundaryElems, enumeration);
        apply_permutation(boundaryCGIDs, enumeration);
        auto firstInvalid =
            std::find(boundaryElems.begin(), boundaryElems.end(), Simplex<D>::invalidSimplex());
        boundaryElems.erase(firstInvalid, boundaryElems.end());
        boundaryCGIDs.erase(boundaryCGIDs.begin() +
                                std::distance(boundaryElems.begin(), firstInvalid),
                            boundaryCGIDs.end());
        boundaryElems.erase(std::unique(boundaryElems.begin(), boundaryElems.end()),
                            boundaryElems.end());
        boundaryCGIDs.erase(std::unique(boundaryCGIDs.begin(), boundaryCGIDs.end()),
                            boundaryCGIDs.end());

        assert(boundaryElems.size() == boundaryCGIDs.size());

        elems.reserve(elems.size() + boundaryElems.size());
        elems.insert(elems.end(), boundaryElems.begin(), boundaryElems.end());
        cGIDs.reserve(cGIDs.size() + boundaryCGIDs.size());
        cGIDs.insert(cGIDs.end(), boundaryCGIDs.begin(), boundaryCGIDs.end());
    }

    // Sort by gid
    std::vector<std::size_t> permutation(elems.size());
    std::iota(permutation.begin(), permutation.end(), std::size_t(0));
    std::sort(permutation.begin(), permutation.end(),
              [&cGIDs](std::size_t a, std::size_t b) { return cGIDs[a] < cGIDs[b]; });
    apply_permutation(elems, permutation);
    apply_permutation(cGIDs, permutation);

    auto lf = LocalFaces<D>(std::move(elems), std::move(cGIDs), numLocal);
    setSharedRanksAndElementData(lf, elemDist);

    // Sort by interior, copy, and ghost
    auto elem2rank = SortedDistributionToRank(elemDist);
    auto const interiorCopyGhostOrder = [&elem2rank, &rank, &lf](std::size_t lid) {
        auto gid = lf.l2cg(lid);
        int p = elem2rank(gid);
        p = rank == p ? -1 : p;
        return std::make_tuple(p, lf.getSharedRanks(lid).size() > 0, gid);
    };
    std::iota(permutation.begin(), permutation.end(), std::size_t(0));
    std::sort(permutation.begin(), permutation.end(),
              [&interiorCopyGhostOrder](std::size_t a, std::size_t b) {
                  return interiorCopyGhostOrder(a) < interiorCopyGhostOrder(b);
              });
    lf.permute(permutation);
    return lf;
}

template <std::size_t D>
void GlobalSimplexMesh<D>::setSharedRanksAndElementData(
    LocalFaces<D>& elems, std::vector<std::size_t> const& elemDist) const {
    int rank, procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    auto elem2rank = SortedDistributionToRank(elemDist);
    std::vector<int> counts(procs, 0);
    for (auto const& c : elems.contiguousGIDs()) {
        ++counts[elem2rank(c)];
    }

    // Exchange data
    AllToAllV a2a(std::move(counts), comm);
    mpi_array_type<uint64_t> mpi_plex_t(D + 1);
    auto requestedElems = a2a.exchange(elems.faces(), mpi_plex_t.get());
    a2a.swap();

    auto [sharedRanks, sharedRanksDispls] = getSharedRanks(requestedElems, a2a);
    elems.setSharedRanks(std::move(sharedRanks), std::move(sharedRanksDispls));

    if (elementData) {
        auto map = makeG2LMap();
        std::vector<std::size_t> lids;
        lids.reserve(requestedElems.size());
        for (auto& elem : requestedElems) {
            auto it = map.find(elem);
            assert(it != map.end());
            lids.emplace_back(it->second);
        }
        elems.setMeshData(elementData->redistributed(lids, a2a));
    }
}

template class GlobalSimplexMesh<2u>;
template class GlobalSimplexMesh<3u>;

} // namespace tndm
