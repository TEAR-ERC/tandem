#ifndef DIETERICHRUINASLIP_20240131_H
#define DIETERICHRUINASLIP_20240131_H

#include "config.h"

#include "geometry/Vector.h"
#include "tensor/Tensor.h"
#include "util/Zero.h"
#include "localoperator/DieterichRuinaAging.h"

#include "mneme/storage.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <iostream>

namespace tndm {

class DieterichRuinaSlip : public DieterichRuinaAging {
public:
    double state_rhs(std::size_t index, double V, double psi) const {
        double myL = p_[index].get<L>();
        return - cp_.b * V / myL * (log(V / cp_.V0) + (psi - cp_.f0) / cp_.b); // Slip law
    }
};

} // namespace tndm

#endif // DIETERICHRUINASLIP_20240131_H