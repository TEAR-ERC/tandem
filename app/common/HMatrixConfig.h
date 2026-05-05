#ifndef HMATRIXCONFIG_20260501_H
#define HMATRIXCONFIG_20260501_H

namespace tndm {

struct HMatrixConfig {
    bool use_hmatrix = false;
    double eta = 0.9;
    int leaf_size = 32;
    int basis_order = 8;
    // max_rank and batch_size are reserved for future MatCreateH2OpusFromMat path
    int max_rank = 64;
    int batch_size = 32;
    double rtol = 1e-4;
};

} // namespace tndm

#endif // HMATRIXCONFIG_20260501_H
