#ifndef SEASELASTICITYADAPTER_20201103_H
#define SEASELASTICITYADAPTER_20201103_H

#include "localoperator/Elasticity.h"
#include "localoperator/Poisson.h"
#include "tandem/SeasAdapterCommon.h"

#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <cstddef>

namespace tndm {

class SeasElasticityAdapter : public SeasAdapterCommon<SeasElasticityAdapter, Elasticity> {
public:
    using SeasAdapterCommon::SeasAdapterCommon;

    TensorBase<Matrix<double>> traction_info() const;
    void traction(std::size_t faultNo, Matrix<double>& traction, LinearAllocator<double>&) const;
    void slip(std::size_t faultNo, Vector<double const>& state, Matrix<double>& s_q) const;
};

} // namespace tndm

#endif // SEASELASTICITYADAPTER_20201103_H
