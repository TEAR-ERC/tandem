#include "MetisPartitioner.h"

#include <parmetis.h>

namespace tndm {

std::vector<idx_t> MetisPartitioner::partition(DistributedCSR<idx_t>& csr, idx_t ncommonnodes,std::vector<idx_t>& elementWeights,
                                               real_t imbalanceTol, MPI_Comm comm) {
    int rank, procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    std::vector<idx_t> partition(csr.dist[rank + 1] - csr.dist[rank]);

    idx_t* elmdist = csr.dist.data();
    idx_t* eptr = csr.rowPtr.data();
    idx_t* eind = csr.colInd.data();
    //idx_t* elmwgt = nullptr;
    idx_t* elmwgt = elementWeights.data(); 
    idx_t wgtflag = 2;
    idx_t numflag = 0;
    idx_t ncon = 1;
    idx_t nparts = procs;
    std::vector<real_t> tpwgt(nparts, 1.0 / nparts);
    real_t* tpwgts = tpwgt.data();
    real_t ubvec = imbalanceTol;
    idx_t options[3] = {1, 0, METIS_RANDOM_SEED};
    idx_t edgecut;
    idx_t* part = partition.data();

    ParMETIS_V3_PartMeshKway(elmdist, eptr, eind, elmwgt, &wgtflag, &numflag, &ncon, &ncommonnodes,
                             &nparts, tpwgts, &ubvec, options, &edgecut, part, &comm);

    return partition;
}

} // namespace tndm
