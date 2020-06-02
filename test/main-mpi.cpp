#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int res = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return res;
}
