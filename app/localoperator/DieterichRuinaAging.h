#ifndef DIETERICHRUINAGING_20241105_H
#define DIETERICHRUINAGING_20241105_H

#include "config.h"

#include "geometry/Vector.h"
#include "localoperator/DieterichRuinaBase.h"
#include "tensor/Tensor.h"
#include "util/Zero.h"

#include "mneme/storage.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <iostream>

namespace tndm {

class DieterichRuinaAging : public DieterichRuinaBase {
public:
    double state_rhs(std::size_t index, double V, double psi) const override {
        double myL = p_[index].get<L>();
        return cp_.b * cp_.V0 / myL * (exp((cp_.f0 - psi) / cp_.b) - V / cp_.V0);
    }
};

} // namespace tndm

#endif // DIETERICHRUINAAGING_20241105_H
