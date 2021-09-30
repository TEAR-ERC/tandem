#ifndef INVERSEINEQUALITY_20210930_H
#define INVERSEINEQUALITY_20210930_H

#include <cstddef>
#include <stdexcept>

namespace tndm {

template <std::size_t D> class InverseInequality {
public:
    /**
     * @brief Inverse trace inequality
     *
     * ||u||_e^2 <= c_N  |e| / |E| ||u||_E^2
     *
     * where e is a facet of simplex E.
     *
     * T. Warburton, J.S. Hesthaven,
     * "On the constants in hp-finite element trace inverse inequalities",
     * Comput. Methods Appl. Mech. Engrg. 192 (2003) 2765--2773,
     * DOI: 10.1016/S0045-7825(03)00294-9
     *
     * @param N Polynomial degree
     *
     * @return c_N
     */
    constexpr static double trace_constant(unsigned N) {
        return (N + 1) * (N + D) / static_cast<double>(D);
    }
    /**
     * @brief Inverse gradient inequality
     *
     * ||grad u||_E^2 <= C_N  |dE|^2 / |E|^2 ||u||_E^2
     *
     * where dE is the boundary of simplex E.
     *
     * Ozisik, Sevtap, Riviere, Beatrice and Warburton, Tim.
     * "On the Constants in Inverse Inequalities in L2." (2010)
     * https://hdl.handle.net/1911/102161
     *
     * @param N Polynomial degree
     *
     * @return C_N
     */
    constexpr static double grad_constant(unsigned N);
};

template <> constexpr double InverseInequality<2>::grad_constant(unsigned N) {

    if (N <= 10) {
        constexpr double C[] = {0.0,      6.0000,   22.5000,  56.8879,   119.8047, 224.1195,
                                385.2210, 620.8674, 951.2557, 1399.0115, 1989.1818};
        return C[N];
    } else {
        throw std::logic_error{"Not implemented"};
        return 0.0;
    }
}

template <> constexpr double InverseInequality<3>::grad_constant(unsigned N) {

    if (N <= 10) {
        constexpr double C[] = {0.0,      10.0000,  31.5000,   73.7490,   148.4089, 269.5513,
                                452.0694, 717.7792, 1085.8205, 1587.8353, 2245.8720};
        return 4.0 * C[N];
    } else {
        throw std::logic_error{"Not implemented"};
        return 0.0;
    }
}

} // namespace tndm

#endif // INVERSEINEQUALITY_20210930_H
