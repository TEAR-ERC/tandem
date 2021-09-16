#include "Summary.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace tndm {

Summary::Summary(double value, MPI_Comm comm) {
    int rank, N;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &N);

    if (rank == 0) {
        auto values = std::vector<double>(N);
        MPI_Gather(&value, 1, MPI_DOUBLE, values.data(), 1, MPI_DOUBLE, 0, comm);

        std::sort(values.begin(), values.end());

        min = values.front();
        if (N % 2 == 1) {
            median = values[N / 2];
        } else {
            median = 0.5 * (values[N / 2 - 1] + values[N / 2]);
        }
        max = values.back();
        sum = std::accumulate(values.begin(), values.end(), 0.0);
        mean = sum / N;
    } else {
        MPI_Gather(&value, 1, MPI_DOUBLE, nullptr, 1, MPI_DOUBLE, 0, comm);
    }
}

} // namespace tndm
