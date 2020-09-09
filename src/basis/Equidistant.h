#ifndef EQUIDISTANT_20200630_H
#define EQUIDISTANT_20200630_H

#include "Nodal.h"

#include <array>
#include <cstddef>
#include <vector>

namespace tndm {

/**
 * @brief Returns equidistant nodes for visualisation purposes.
 */
template <std::size_t D> class EquidistantNodesFactory : public NodesFactory<D> {
public:
    std::vector<std::array<double, D>> operator()(unsigned degree) const override;

private:
    template <std::size_t DD>
    std::array<std::array<double, D>, DD + 1>
    shrinkAndShift(double factor, std::array<std::array<double, D>, DD + 1> const& verts) const;

    void edge(int n, std::array<std::array<double, D>, 2> const& verts,
              std::vector<std::array<double, D>>& result) const;

    void triangle(int n, std::array<std::array<double, D>, 3>&& verts,
                  std::vector<std::array<double, D>>& result) const;

    void tet(int n, std::array<std::array<double, D>, 4>&& verts,
             std::vector<std::array<double, D>>& result) const;
};

} // namespace tndm

#endif // EQUIDISTANT_20200630_H
