#ifndef SUMMARY_20210916_H
#define SUMMARY_20210916_H

#include <mpi.h>

namespace tndm {

struct Summary {
    Summary(double value, MPI_Comm comm);

    double min;
    double median;
    double mean;
    double max;
    double sum;
};

} // namespace tndm

#endif // SUMMARY_20210916_H
