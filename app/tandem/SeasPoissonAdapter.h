#ifndef SEASPOISSONADAPTER_20201102_H
#define SEASPOISSONADAPTER_20201102_H

#include "localoperator/Poisson.h"
#include "tandem/SeasAdapterCommon.h"

#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <cstddef>

namespace tndm {

class SeasPoissonAdapter : public SeasAdapterCommon<SeasPoissonAdapter, Poisson> {
public:
    using SeasAdapterCommon::SeasAdapterCommon;

    TensorBase<Matrix<double>> traction_info() const;
    void traction(std::size_t faultNo, Matrix<double>& traction, LinearAllocator<double>&) const;
    void slip(std::size_t faultNo, Vector<double const>& state, Matrix<double>& s_q) const;
};

} // namespace tndm

#endif // SEASPOISSONADAPTER_20201102_H
