#ifndef NUMBERINGCONVENTION_20211118_H
#define NUMBERINGCONVENTION_20211118_H

#include <array>
#include <type_traits>

namespace tndm {

/**
 * @brief Numbering convention for simplices
 *
 * VTK:
 * https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
 * GMSH: https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
 */
enum class NumberingConvention { VTK = 0, GMSH = 1 };

constexpr static std::array<std::array<std::array<unsigned, 2>, 6>, 2> TetEdgeConventions = {{
    {{{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}}}, // VTK
    {{{0, 1}, {1, 2}, {2, 0}, {3, 0}, {3, 2}, {3, 1}}}  // GMSH
}};

constexpr static std::array<std::array<std::array<unsigned, 3>, 4>, 2> TetFaceConventions = {{
    {{{0, 1, 3}, {2, 3, 1}, {0, 3, 2}, {0, 2, 1}}}, // VTK
    {{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {3, 1, 2}}}  // GMSH
}};

constexpr auto const& tet_edge_convention(NumberingConvention convention) {
    return TetEdgeConventions[static_cast<std::underlying_type_t<NumberingConvention>>(convention)];
}

constexpr auto const& tet_face_convention(NumberingConvention convention) {
    return TetFaceConventions[static_cast<std::underlying_type_t<NumberingConvention>>(convention)];
}

} // namespace tndm

#endif // NUMBERINGCONVENTION_20211118_H
