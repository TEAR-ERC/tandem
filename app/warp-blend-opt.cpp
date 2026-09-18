#include "basis/Nodal.h"
#include "basis/WarpAndBlend.h"

#include <argparse.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using tndm::LebesgueFunction;
using tndm::warpAndBlendAlpha;
using tndm::WarpAndBlendFactory;

template <typename Func>
double GoldenSectionSearch(Func f, double a, double b, double tol = 1.0e-5) {
    const double invGR = 2.0 / (sqrt(5.0) + 1.0);
    double h = b - a;
    std::array<double, 4> x{a, b - h * invGR, a + h * invGR, b};
    std::array<double, 2> fx{f(x[1]), f(x[2])};
    while (fabs(h) > tol) {
        if (fx[0] <= fx[1]) {
            h = x[2] - x[0];
            x = {x[0], x[2] - h * invGR, x[1], x[2]};
            fx = {f(x[1]), fx[0]};
        } else {
            h = x[3] - x[1];
            x = {x[1], x[2], x[1] + h * invGR, x[3]};
            fx = {fx[1], f(x[2])};
        }
    }
    return 0.5 * (x[0] + x[3]);
}

template <std::size_t D>
std::array<double, D> sample(std::mt19937& gen, std::uniform_real_distribution<>& dis);

template <>
std::array<double, 2> sample<2>(std::mt19937& gen, std::uniform_real_distribution<>& dis) {
    double x = dis(gen);
    double y = dis(gen);
    if (x + y > 1.0) {
        return {1.0 - y, 1.0 - x};
    }
    return {x, y};
}

template <>
std::array<double, 3> sample<3>(std::mt19937& gen, std::uniform_real_distribution<>& dis) {
    double x = dis(gen);
    double y = dis(gen);
    double z = dis(gen);
    if (x + y > 1) {
        x = 1.0 - x;
        y = 1.0 - y;
    }
    if (x + y + z > 1.0) {
        if (y + z > 1.0) {
            return {x, 1.0 - z, 1.0 - y - x};
        } else {
            return {1 - y - z, y, x + y + z - 1.0};
        }
    }
    return {x, y, z};
}

template <std::size_t D> using RandomSample = std::vector<std::array<double, D>>;

template <std::size_t D> RandomSample<D> generateSample(int sampleSize) {
    RandomSample<D> result(sampleSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < sampleSize; ++i) {
        result[i] = sample<D>(gen, dis);
    }

    return result;
}

template <std::size_t D>
double estimateLebesgueConstant(unsigned degree, std::vector<std::array<double, D>> const& nodes,
                                std::vector<std::array<double, D>> const& positions) {
    double L = 0.0;
#pragma omp parallel shared(L)
    {
        LebesgueFunction<D> lb(degree, nodes);

#pragma omp for reduction(max : L)
        for (std::size_t i = 0; i < positions.size(); ++i) {
            L = std::max(L, lb(positions[i]));
        }
    }
    return L;
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("Test");
    program.add_argument("-a", "--alpha")
        .help("alpha parameter in warp & blend (0 <= alpha <= 2, supply negative alpha for "
              "tabulated value)")
        .default_value(-1.0)
        .action([](std::string const& value) { return std::stod(value); });
    program.add_argument("-D", "--dim")
        .help("Simplex dimension (D=2: triangle, D=3: tet)")
        .default_value(2)
        .action([](std::string const& value) { return std::stoi(value); });
    program.add_argument("-p", "--plot")
        .help("Evaluate alpha at P points")
        .default_value(0)
        .action([](std::string const& value) { return std::stoi(value); });
    program.add_argument("-o", "--optimize")
        .help("Optimize alpha parameter")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-t", "--tolerance")
        .help("Tolerance used in optimisation")
        .default_value(1.0e-5)
        .action([](std::string const& value) { return std::stod(value); });
    program.add_argument("-s", "--sampleSize")
        .help("Random sample size used to estimate Lebesgue constant")
        .default_value(200000)
        .action([](std::string const& value) { return std::stoi(value); });
    program.add_argument("--csv")
        .help("Print in CSV format")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--csv-headless")
        .help("Print in CSV without header")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("degree")
        .help("polynomial degrees")
        .remaining()
        .action([](std::string const& value) { return std::stoi(value); });

    try {
        program.parse_args(argc, argv);
    } catch (std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 0;
    }

    const auto alpha = program.get<double>("-a");
    const auto D = program.get<int>("-D");
    const auto sampleSize = program.get<int>("-s");
    const auto plot = program.get<int>("-p");
    const auto optimize = program.get<bool>("-o");
    const auto tolerance = program.get<double>("-t");
    const auto csvHeadless = program.get<bool>("--csv-headless");
    const auto csv = csvHeadless || program.get<bool>("--csv");
    const auto degrees = program.get<std::vector<int>>("degree");

    if (!csv) {
        std::cout << "Parameters:" << std::endl;
        std::cout << "\tSimplex dimension: " << D << std::endl;
        std::cout << "\tRandom sample size: " << sampleSize << std::endl;
        if (optimize) {
            std::cout << "\tTolerance: " << tolerance << std::endl;
        }
        std::cout << "\tPolynomial degrees:";
        for (auto& degree : degrees) {
            std::cout << " " << degree;
        }
        std::cout << std::endl;
    } else if (!csvHeadless) {
        std::cout << "D,degree,alpha,L" << std::endl;
    }

    RandomSample<2> sample2;
    RandomSample<3> sample3;
    std::function<double(int, double)> lebesgueConstant;
    if (D == 2) {
        sample2 = generateSample<2>(sampleSize);
        lebesgueConstant = [&sample2](int degree, double alpha) {
            auto factory = WarpAndBlendFactory<2>([alpha](unsigned) { return alpha; });
            return estimateLebesgueConstant(degree, factory(degree), sample2);
        };
    } else if (D == 3) {
        sample3 = generateSample<3>(sampleSize);
        lebesgueConstant = [&sample3](int degree, double alpha) {
            auto factory = WarpAndBlendFactory<3>([alpha](unsigned) { return alpha; });
            return estimateLebesgueConstant(degree, factory(degree), sample3);
        };
    } else {
        std::cerr << "Test for simplex dimension " << D << " is not implemented." << std::endl;
        return -1;
    }

    auto printResult = [&csv, &D](int degree, double alpha, double L) {
        if (csv) {
            std::cout << D << "," << degree << "," << alpha << "," << L << std::endl;
        } else {
            std::cout << "\tL_" << degree << "(" << alpha << ") = " << L << std::endl;
        }
    };

    constexpr double alphaMax = 2.0;

    for (auto& degree : degrees) {
        if (!csv) {
            std::cout << "N = " << degree << ":" << std::endl;
        }

        if (plot > 0) {
            double ha = alphaMax / plot;
            for (int i = 0; i <= plot; ++i) {
                double myAlpha = i * ha;
                auto L = lebesgueConstant(degree, myAlpha);
                printResult(degree, myAlpha, L);
            }
        } else {
            double myAlpha = (alpha < 0.0) ? warpAndBlendAlpha(D, degree) : alpha;
            if (optimize) {
                auto lbc = [&degree, &lebesgueConstant](double alpha) {
                    return lebesgueConstant(degree, alpha);
                };
                myAlpha = GoldenSectionSearch(lbc, 0.0, alphaMax, tolerance);
            }
            auto L = lebesgueConstant(degree, myAlpha);
            printResult(degree, myAlpha, L);
        }
    }

    return 0;
}
