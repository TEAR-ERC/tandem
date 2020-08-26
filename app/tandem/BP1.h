#ifndef BP1_20200820_H
#define BP1_20200820_H

#include "config.h"

#include <array>

namespace tndm {

class BP1 {
public:
    using point_t = std::array<double, DomainDimension>;

    void setX(point_t x);

    double F(double tau, double V, double psi) const;
    double dFdV(double V, double psi) const;
    std::array<double, 2> iniState(double V, double psi, double s_n, double eta) const;
    double tau0() const;
    double psi0() const;

    double computeSlipRate(double tau, double psi) const;
    double G(double tau, double V, double psi) const { return 1.0 - V * psi / L_; }

private:
    double a_, tau0_;

    constexpr static double a0_ = 0.010;
    constexpr static double amax_ = 0.025;
    constexpr static double H_ = 15e3;
    constexpr static double h_ = 3e3;
    constexpr static double b_ = 0.015;
    constexpr static double V0_ = 1.0e-6;
    constexpr static double Vinit_ = 1.0e-9;
    constexpr static double L_ = 0.008;
    constexpr static double f0_ = 0.6;
    // constexpr static double sn_ = 50e6;
    constexpr static double sn_ = 1e6;
    // constexpr static double sn_ = 50;
    constexpr static double eta_ = 2670.0 * 3464.0 / 2.0;
    // constexpr static double eta_ = 2.670 * 3.464 / 2.0;
    constexpr static std::size_t maxIts_ = 100;
};

} // namespace tndm

#endif // BP1_20200820_H
