#include "GlobalSimplexMesh.h"
#include "parallel/MetisPartitioner.h"
#include "parallel/SortedDistribution.h"
#include "util/Algorithm.h"

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
    if (volumeData) {
        volumeData = volumeData->redistributed(enumeration, a2a);
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
        auto owner = std::vector<int>(cGIDs.size());
        std::fill(owner.begin(), owner.end(), rank);
        auto lf = LocalFaces<D>(std::move(elems), std::move(owner), std::move(cGIDs), elems.size());
        setSharedRanksAndElementData(lf, elemDist);
        return lf;
    }

    mpi_array_type<uint64_t> mpi_facet_t(D);
    mpi_array_type<uint64_t> mpi_elem_t(D + 1u);
    auto plex2rank = getPlex2Rank<D - 1u>();

    auto const sortInteriorCopy = [this](auto& elems, unsigned overlap) {
        auto up = makeUpwardMap(elems);
        auto boundaryFaces = getBoundaryFaces(up);
        deleteDomainBoundaryFaces(boundaryFaces);
        std::unordered_set<std::size_t> copyElNos;
        for (auto&& face : boundaryFaces) {
            auto [it, end] = up.equal_range(face);
            for (; it != end; ++it) {
                copyElNos.insert(it->second);
            }
        }
        for (unsigned level = 1; level < overlap; ++level) {
            std::unordered_set<std::size_t> newCopyElNos;
            for (auto&& elNo : copyElNos) {
                auto dws = elems[elNo].template downward<D - 1u>();
                for (auto&& dw : dws) {
                    auto [it, end] = up.equal_range(dw);
                    for (; it != end; ++it) {
                        if (copyElNos.find(it->second) == copyElNos.end()) {
                            newCopyElNos.insert(it->second);
                        }
                    }
                }
            }
            copyElNos.insert(newCopyElNos.begin(), newCopyElNos.end());
        }
        auto elNos = std::vector<std::size_t>(elems.size());
        std::iota(elNos.begin(), elNos.end(), 0ul);
        std::sort(elNos.begin(), elNos.end(), [&copyElNos](std::size_t a, std::size_t b) {
            auto copyA = copyElNos.find(a) != copyElNos.end() ? 1 : 0;
            auto copyB = copyElNos.find(b) != copyElNos.end() ? 1 : 0;
            return std::make_pair(copyA, a) < std::make_pair(copyB, b);
        });
        apply_permutation(elems, std::move(elNos));
        return elems.size() - copyElNos.size();
    };
    std::size_t copyBegin = sortInteriorCopy(elems, overlap);

    auto const makeDistributedUpwardMap = [&procs, &myComm, &mpi_facet_t, &mpi_elem_t,
                                           &plex2rank](upward_map_t const& up, auto&& elems,
                                                       std::vector<std::size_t> const& cGIDs,
                                                       std::size_t copyBegin) {
        facet_set_t requiredFaces;
        for (std::size_t elNo = copyBegin; elNo < elems.size(); ++elNo) {
            auto const dws = elems[elNo].template downward<D - 1u>();
            for (auto const& dw : dws) {
                requiredFaces.insert(dw);
            }
        }
        std::vector<int> counts(procs, 0);
        for (auto&& face : requiredFaces) {
            counts[plex2rank(face)] += up.count(face);
        }
        std::vector<std::ptrdiff_t> offsets(procs + 1);
        offsets[0] = 0;
        std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);

        std::vector<Simplex<D - 1u>> upFaces(offsets.back());
        std::vector<Simplex<D>> upElems(offsets.back());
        std::vector<std::size_t> upCGIDs(offsets.back());
        for (auto&& face : requiredFaces) {
            auto rank = plex2rank(face);
            auto [it, end] = up.equal_range(face);
            for (; it != end; ++it) {
                upFaces[offsets[rank]] = face;
                upElems[offsets[rank]] = elems[it->second];
                upCGIDs[offsets[rank]] = cGIDs[it->second];
                ++offsets[rank];
            }
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

    auto upMap = makeUpwardMap(elems);
    auto distUp = makeDistributedUpwardMap(upMap, elems, cGIDs, copyBegin);

    std::size_t numLocal = elems.size();

    for (int level = 0; level < static_cast<int>(overlap); ++level) {
        auto boundaryFaces = getBoundaryFaces(upMap);

        auto const getRequestedFacesAndMissingVertices =
            [&procs, &myComm, &mpi_facet_t, &plex2rank](auto&& elems, upward_map_t const& upMap,
                                                        facet_set_t const& boundaryFaces) {
                std::vector<int> counts(procs, 0);
                for (auto&& face : boundaryFaces) {
                    ++counts[plex2rank(face)];
                }
                std::vector<std::ptrdiff_t> offsets(procs + 1);
                offsets[0] = 0;
                std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);

                std::vector<Simplex<D - 1u>> boundaryFacesVec(offsets.back());
                std::vector<uint64_t> missingVertex(offsets.back());
                for (auto&& face : boundaryFaces) {
                    assert(upMap.count(face) == 1);
                    // Find missing vertex i.e. elem \ facet
                    auto& elem = elems[upMap.find(face)->second];
                    auto f = face.begin();
                    auto e = elem.begin();
                    while (e != elem.end() && f != face.end() && *e == *f) {
                        ++f;
                        ++e;
                    }
                    assert(e != elem.end());

                    auto rank = plex2rank(face);
                    boundaryFacesVec[offsets[rank]] = face;
                    missingVertex[offsets[rank]] = *e;
                    ++offsets[rank];
                }

                AllToAllV a2a(std::move(counts), myComm);
                auto requestedBoundaryFaces = a2a.exchange(boundaryFacesVec, mpi_facet_t.get());
                auto requestedMissingVertices = a2a.exchange(missingVertex);
                return std::make_tuple(requestedBoundaryFaces, requestedMissingVertices, a2a);
            };

        auto [requestedBoundaryFaces, requestedMissingVertices, a2a] =
            getRequestedFacesAndMissingVertices(elems, upMap, boundaryFaces);

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

        auto oldSize = elems.size();
        elems.reserve(oldSize + boundaryElems.size());
        elems.insert(elems.end(), boundaryElems.begin(), boundaryElems.end());
        cGIDs.reserve(cGIDs.size() + boundaryCGIDs.size());
        cGIDs.insert(cGIDs.end(), boundaryCGIDs.begin(), boundaryCGIDs.end());

        for (std::size_t elNo = oldSize; elNo < elems.size(); ++elNo) {
            auto downward = elems[elNo].template downward<D - 1u>();
            for (auto& s : downward) {
                upMap.emplace(s, elNo);
            }
        }
    }

    // Sort by gid
    auto gid2owner = SortedDistributionToRank(elemDist);
    auto const intCopyGhostOrder = [&rank, &gid2owner, &cGIDs](std::size_t lid) {
        auto gid = cGIDs[lid];
        int p = gid2owner(gid);
        p = rank == p ? -1 : p;
        return std::make_pair(p, gid);
    };
    std::vector<std::size_t> permutation(elems.size());
    std::iota(permutation.begin(), permutation.end(), std::size_t(0));
    std::sort(permutation.begin(), permutation.end(),
              [&intCopyGhostOrder](std::size_t a, std::size_t b) {
                  return intCopyGhostOrder(a) < intCopyGhostOrder(b);
              });
    apply_permutation(elems, permutation);
    apply_permutation(cGIDs, permutation);

    auto owner = std::vector<int>(cGIDs.size());
    std::transform(cGIDs.begin(), cGIDs.end(), owner.begin(), gid2owner);

    auto lf = LocalFaces<D>(std::move(elems), std::move(owner), std::move(cGIDs), numLocal);
    setSharedRanksAndElementData(lf, elemDist);

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
    auto rankPerm = std::vector<int>(procs);
    rankPerm[0] = rank;
    std::iota(rankPerm.begin() + 1, rankPerm.begin() + 1 + rank, 0);
    std::iota(rankPerm.begin() + 1 + rank, rankPerm.end(), rank + 1);
    AllToAllV a2a(std::move(counts), comm);
    a2a.setRankPermutation(rankPerm);
    mpi_array_type<uint64_t> mpi_plex_t(D + 1);
    auto requestedElems = a2a.exchange(elems.faces(), mpi_plex_t.get());
    a2a.swap();

    auto [sharedRanks, sharedRanksDispls] = getSharedRanks(requestedElems, a2a, &rankPerm);
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
    if (volumeData) {
        auto map = makeG2LMap();
        std::vector<std::size_t> lids;
        lids.reserve(requestedElems.size());
        for (auto& elem : requestedElems) {
            auto it = map.find(elem);
            assert(it != map.end());
            lids.emplace_back(it->second);
        }
        elems.setVolumeData(volumeData->redistributed(lids, a2a));
    }
}

template <std::size_t D>
void GlobalSimplexMesh<D>::deleteDomainBoundaryFaces(facet_set_t& boundaryFaces) const {
    int procs;
    MPI_Comm_size(comm, &procs);
    auto myComm = comm;
    auto plex2rank = getPlex2Rank<D - 1u>();
    auto const figureOutWhichFacesAppearTwiceInDistributedMemory = [&procs, &myComm, &plex2rank](
                                                                       facet_set_t const& bfaces) {
        std::vector<int> counts(procs, 0);
        for (auto const& u : bfaces) {
            ++counts[plex2rank(u)];
        }
        // Send domain and partition boundary faces to face owner
        std::vector<std::ptrdiff_t> offsets(procs + 1);
        offsets[0] = 0;
        std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);
        auto faces = std::vector<Simplex<D - 1u>>(offsets.back());
        for (auto const& u : bfaces) {
            auto rank = plex2rank(u);
            faces[offsets[rank]++] = u;
        }
        AllToAllV a2a(std::move(counts), myComm);
        mpi_array_type<uint64_t> mpi_plex_t(D);
        auto requestedFaces = a2a.exchange(faces, mpi_plex_t.get());
        auto requestedFacesAsMultiset =
            std::unordered_multiset<Simplex<D - 1u>, SimplexHash<D - 1u>>(requestedFaces.begin(),
                                                                          requestedFaces.end());
        auto requestedFaceCount = std::vector<std::size_t>(requestedFaces.size());
        for (std::size_t fNo = 0; fNo < requestedFaces.size(); ++fNo) {
            requestedFaceCount[fNo] = requestedFacesAsMultiset.count(requestedFaces[fNo]);
        }
        a2a.swap();
        auto faceCount = a2a.exchange(requestedFaceCount);
        return std::make_pair(faces, faceCount);
    };

    auto [faces, faceCount] = figureOutWhichFacesAppearTwiceInDistributedMemory(boundaryFaces);

    assert(faceCount.size() == faces.size());
    for (std::size_t fNo = 0; fNo < faceCount.size(); ++fNo) {
        assert(1 <= faceCount[fNo] && faceCount[fNo] <= 2);
        if (faceCount[fNo] == 1) {
            assert(boundaryFaces.find(faces[fNo]) != boundaryFaces.end());
            boundaryFaces.erase(faces[fNo]);
        }
    }
}

template class GlobalSimplexMesh<1u>;
template class GlobalSimplexMesh<2u>;
template class GlobalSimplexMesh<3u>;

} // namespace tndm
