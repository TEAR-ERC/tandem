#ifndef POISSONADAPTER_20210906_H
#define POISSONADAPTER_20210906_H

#include "localoperator/AdapterBase.h"
#include "localoperator/Poisson.h"

#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <cstddef>

namespace tndm {

class PoissonAdapter : public AdapterBase {
public:
    using AdapterBase::AdapterBase;
    using local_operator_t = Poisson;

    std::size_t traction_block_size() const;
    void traction(std::size_t faultNo, Matrix<double> const& traction_q, Vector<double>& traction,
                  LinearAllocator<double>&) const;
    void slip(std::size_t faultNo, Vector<double const>& state, Matrix<double>& s_q) const;
};

} // namespace tndm

#endif // POISSONADAPTER_20210906_H
