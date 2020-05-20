#ifndef GLOBALSIMPLEXMESH_H
#define GLOBALSIMPLEXMESH_H

#include "LocalFaces.h"
#include "LocalSimplexMesh.h"
#include "Simplex.h"

#include "parallel/CommPattern.h"
#include "parallel/DistributedCSR.h"
#include "parallel/MPITraits.h"
#include "parallel/MetisPartitioner.h"
#include "parallel/SortedDistribution.h"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tndm {

template <std::size_t D> class GlobalSimplexMesh {
public:
    using simplex_t = Simplex<D>;
    static_assert(sizeof(simplex_t) == (D+1)*sizeof(int));

    GlobalSimplexMesh(std::size_t numVertices, std::vector<simplex_t>&& elements,
                      MPI_Comm comm = MPI_COMM_WORLD)
        : numVerts(numVertices), elems(std::move(elements)), comm(comm) {
        vtxdist = makeSortedDistribution(numVerts);
    }

    std::size_t numVertices() const { return numVerts; }
    std::size_t numElements() const { return elems.size(); }

    template<typename OutIntT>
    DistributedCSR<OutIntT> distributedCSR() const {
        DistributedCSR<OutIntT> csr;

        auto elmdist = makeSortedDistribution(numElements());
        csr.dist.reserve(elmdist.size());
        std::copy(elmdist.begin(), elmdist.end(), csr.dist.begin());

        auto numElems = numElements();
        csr.rowPtr.resize(numElems+1);
        csr.colInd.resize(numElems*(D+1));

        OutIntT ind = 0;
        OutIntT ptr = 0;
        for (auto& e : elems) {
            csr.rowPtr[ptr++] = ind;
            for (auto& p : e) {
                csr.colInd[ind++] = p;
            }
        }
        csr.rowPtr.back() = ind;

        return csr;
    }

    void repartition() {
        int procs, rank;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &procs);

        auto distCSR = distributedCSR<idx_t>();
        auto partition = MetisPartitioner::partition(distCSR, D);
        assert(partition.size() == numElements());

        std::vector<idx_t> enumeration(partition.size());
        std::iota(enumeration.begin(), enumeration.end(), 0);

        std::sort(enumeration.begin(), enumeration.end(), [&partition](auto a, auto b) {
            return partition[a] < partition[b];
        });

        std::vector<int> sendcounts(procs, 0);
        std::vector<simplex_t> elemsToSend;
        elemsToSend.reserve(numElements());
        auto eIt = enumeration.begin();
        for (int p = 0; p < procs; ++p) {
            while (partition[*eIt] <= p && eIt != enumeration.end()) {
                ++sendcounts[p];
                elemsToSend.emplace_back(elems[*eIt]);
                ++eIt;
            }
        }
        assert(eIt == enumeration.end());

        AllToAllV a2a(std::move(sendcounts));
        mpi_array_type<int> mpi_simplex_t(D+1);
        elems = a2a.exchange(elemsToSend, mpi_simplex_t.get());
    }

    LocalSimplexMesh<D> getLocalMesh() const {
        auto localFaces = getAllLocalFaces(std::make_index_sequence<D>{});

        return LocalSimplexMesh<D>(std::move(localFaces));
    }

    template <std::size_t DD> int simplex2Location(Simplex<DD> const& plex) const {
        if constexpr (DD == 0) {
            int rank;
            MPI_Comm_rank(comm, &rank);
            assert(plex[0] >= vtxdist[rank] && plex[0] < vtxdist[rank + 1]);
            return plex[0] - vtxdist[rank];
        }
        assert(false);
        return 0;
    }

private:
    template <std::size_t... Is> auto getAllLocalFaces(std::index_sequence<Is...>) const {
        std::vector<Simplex<D>> elemsCopy(elems);
        return std::make_tuple(getFaces<Is>()..., LocalFaces<D>(std::move(elemsCopy)));
    }

    template <std::size_t DD> auto getPlex2Rank() const {
        if constexpr (DD == 0) {
            SortedDistributionToRank v2r(vtxdist);
            return [v2r](Simplex<0> const& plex) { return v2r(plex[0]); };
        } else {
            int procs;
            MPI_Comm_size(comm, &procs);
            return [procs](Simplex<DD> const& plex) { return SimplexHash<DD>()(plex) % procs; };
        }
    }

    template <std::size_t DD> auto getFaces() const {
        auto plex2rank = getPlex2Rank<DD>();

        int procs;
        MPI_Comm_size(comm, &procs);

        std::vector<std::set<Simplex<DD>>> requiredFaces(procs);
        for (auto& elem : elems) {
            auto downward = elem.template downward<DD>();
            for (auto& s : downward) {
                requiredFaces[plex2rank(s)].insert(s);
            }
        }
        std::vector<int> counts(procs, 0);
        std::size_t total = 0;
        for (int p = 0; p < procs; ++p) {
            auto size = requiredFaces[p].size();
            counts[p] = size;
            total += size;
        }

        std::vector<Simplex<DD>> faces;
        faces.reserve(total);
        for (auto& perRank : requiredFaces) {
            std::copy(perRank.begin(), perRank.end(), std::back_inserter(faces));
        }

        return LocalFaces<DD>(std::move(faces), Displacements(counts));
    }

    std::size_t numVerts;
    std::vector<simplex_t> elems;
    MPI_Comm comm;
    std::vector<std::size_t> vtxdist;
};
}


#endif // GLOBALSIMPLEXMESH_H
