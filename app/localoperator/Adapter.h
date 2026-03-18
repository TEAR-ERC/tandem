#ifndef ADAPTER_20210910_H
#define ADAPTER_20210910_H

#include "localoperator/AdapterBase.h"

#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <cstddef>

namespace tndm {

template <typename T> class Adapter : public AdapterBase {
public:
    using AdapterBase::AdapterBase;
    using local_operator_t = T;

    std::size_t traction_block_size() const;
    void traction(std::size_t faultNo, Matrix<double> const& traction_q, Vector<double>& traction,
                  LinearAllocator<double>&) const;
    void slip(std::size_t faultNo, Vector<double const>& state, Matrix<double>& s_q) const;
    void slip_rate(std::size_t faultNo, Vector<double const>& state,
                   Matrix<double>& slip_rate_q) const;
    void moment_rate(std::size_t faultNo, Matrix<double>& moment_rate_vector,
                     Matrix<double>& slip_rate_q, Matrix<double>& mu_field) const;
};

} // namespace tndm

#endif // ADAPTER_20210910_H
