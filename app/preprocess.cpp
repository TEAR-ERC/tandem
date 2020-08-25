#include "quadrules/AutoRule.h"
#include "util/Combinatorics.h"

#include <argparse.hpp>

#include <fstream>
#include <iostream>
#include <string>

template <std::size_t D>
void preprocess(unsigned N, unsigned minQuadOrder, std::string const& outputFileName) {
    auto fctRule = tndm::simplexQuadratureRule<D - 1u>(minQuadOrder);
    auto elRule = tndm::simplexQuadratureRule<D>(minQuadOrder);

    std::ofstream file;
    file.open(outputFileName);
    file << "{\n"
         << "  \"dim\": " << D << ",\n"
         << "  \"degree\": " << N << ",\n"
         << "  \"numFacetBasisFunctions\": " << tndm::binom(N + D - 1u, D - 1u) << ",\n"
         << "  \"numElementBasisFunctions\": " << tndm::binom(N + D, D) << ",\n"
         << "  \"numFacetQuadPoints\": " << fctRule.size() << ",\n"
         << "  \"numElementQuadPoints\": " << elRule.size() << "\n"
         << "}\n";
    file.close();
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("preprocess");
    program.add_argument("-Q")
        .help("Minimum quadrature order")
        .default_value(0u)
        .action([](std::string const& value) { return static_cast<unsigned>(std::stoul(value)); });
    program.add_argument("-o").help("Output file name").default_value(std::string("options.json"));
    program.add_argument("D")
        .help("Simplex dimension (D=2: triangle, D=3: tet)")
        .action([](std::string const& value) { return static_cast<unsigned>(std::stoul(value)); });
    program.add_argument("N").help("Polynomial degree").action([](std::string const& value) {
        return static_cast<unsigned>(std::stoul(value));
    });

    try {
        program.parse_args(argc, argv);
    } catch (std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 0;
    }

    auto outputFileName = program.get("-o");
    auto D = program.get<unsigned>("D");
    auto N = program.get<unsigned>("N");
    auto minQuadOrder = program.get<unsigned>("-Q");
    if (minQuadOrder == 0u) {
        minQuadOrder = 2u * N + 1u;
    }

    switch (D) {
    case 2u:
        preprocess<2u>(N, minQuadOrder, outputFileName);
        break;
    case 3u:
        preprocess<3u>(N, minQuadOrder, outputFileName);
        break;
    default:
        std::cerr << "Unsupported dimension " << D << std::endl;
        return -1;
    }

    return 0;
}
