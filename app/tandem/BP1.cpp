#include "BP1.h"

#include "util/Zero.h"

#include <cmath>

namespace tndm {

void BP1::setX(point_t x) {
    double const z = -x[1];
    if (z < H_) {
        a_ = a0_;
    } else if (z < H_ + h_) {
        a_ = a0_ + (amax_ - a0_) * (z - H_) / h_;
    } else {
        a_ = amax_;
    }
}

double BP1::F(double tau, double V, double psi) const {
    double e = exp((f0_ + b_ * log(V0_ * psi / L_)) / a_);
    double f = a_ * asinh((V / (2.0 * V0_)) * e);
    return tau + tau0() - sn_ * f - eta_ * V;
}

double BP1::dFdV(double V, double psi) const {
    double e = exp((f0_ + b_ * log(V0_ * psi / L_)) / a_);
    double dasinh = 1.0 / sqrt(V * V * e * e / (4.0 * V0_ * V0_) + 1.0);
    double dfdV = a_ * dasinh * e / (2.0 * V0_);
    return -sn_ * dfdV - eta_;
}

double BP1::tau0() const {
    double e = exp((f0_ + b_ * log(V0_ / Vinit_)) / amax_);
    return sn_ * amax_ * asinh((Vinit_ / (2.0 * V0_)) * e) + eta_ * Vinit_;
}

double BP1::psi0() const {
    double s = sinh((tau0() - eta_ * Vinit_) / (a_ * sn_));
    double l = log((2.0 * V0_ / Vinit_) * s);
    return (L_ / V0_) * exp((a_ / b_) * l - f0_ / b_);
}

double BP1::computeSlipRate(double tau, double psi) const {
    double tauAbs = tau + tau0();
    double a = -tauAbs / eta_;
    double b = tauAbs / eta_;
    auto fF = [this, &tau, &psi](double V) { return this->F(tau, V, psi); };
    return zeroIn(a, b, fF);
}

} // namespace tndm
