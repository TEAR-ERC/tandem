#ifndef SEASSOLUTION_20210719_H
#define SEASSOLUTION_20210719_H

#include "config.h"

#include "script/LuaLib.h"
#include "tensor/Tensor.h"

#include <array>
#include <cstddef>

namespace tndm {

template <std::size_t NumQuantities> class SeasSolution {
public:
    using time_functional_t = LuaLib::functional_t<DomainDimension + 1, NumQuantities>;

    SeasSolution(time_functional_t solution) : solution_(std::move(solution)) {}

    std::array<double, NumQuantities> operator()(Vector<double> const& v) const {
        std::array<double, DomainDimension + 1> xt;
        for (std::size_t i = 0; i < DomainDimension; ++i) {
            xt[i] = v(i);
        }
        xt.back() = time_;
        return solution_(xt);
    }

    void set_time(double time) { time_ = time; }

private:
    time_functional_t solution_;
    double time_ = 0.0;
};

} // namespace tndm

#endif // SEASSOLUTION_20210719_H
