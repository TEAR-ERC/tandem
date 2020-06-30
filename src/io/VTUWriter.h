#ifndef VTUWRITER_20200629_H
#define VTUWRITER_20200629_H

#include "basis/Equidistant.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "tensor/Tensor.h"

#include <tinyxml2.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tndm {

template <std::size_t D> class VTUWriter {
public:
    static constexpr std::size_t PointDim = 3u;
    using header_t = uint64_t;

    static int32_t VTKType(bool linear);

    VTUWriter(unsigned degree = 1u) : refNodes_(EquidistantNodesFactory<D>()(degree)) {
        auto grid = doc_.NewElement("UnstructuredGrid");
        doc_.InsertFirstChild(grid);
    }

    void addMesh(Curvilinear<D>& cl);
    void addData(std::string const& name, FiniteElementFunction<D> const& function);
    bool write(std::string const& baseName);

private:
    template <typename T>
    tinyxml2::XMLElement* addDataArray(tinyxml2::XMLElement* parent, std::string const& name,
                                       std::size_t inComponents, std::size_t outComponents,
                                       std::vector<T> const& data);

    std::vector<std::array<double, D>> refNodes_;
    std::vector<unsigned char> appended_;
    tinyxml2::XMLDocument doc_;
};

} // namespace tndm

#endif // VTUWRITER_20200629_H
