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

template <std::size_t D> class VTUPiece;

template <std::size_t D> class VTUWriter {
public:
    static constexpr std::size_t PointDim = 3u;
    using header_t = uint64_t;

    static int32_t VTKType(bool linear);

    VTUWriter(unsigned degree = 1u) : refNodes_(EquidistantNodesFactory<D>()(degree)) {
        auto grid = doc_.NewElement("UnstructuredGrid");
        doc_.InsertFirstChild(grid);
    }

    VTUPiece<D> addPiece(Curvilinear<D>& cl);

    /**
     * @brief Write VTU to disk.
     *
     * @param baseName File name without extension
     *
     * @return True if write was successful.
     */
    bool write(std::string const& baseName);

    std::vector<std::array<double, D>> const& refNodes() const { return refNodes_; }

private:
    friend class VTUPiece<D>;

    template <typename T>
    tinyxml2::XMLElement* addDataArray(tinyxml2::XMLElement* parent, std::string const& name,
                                       std::size_t inComponents, std::size_t outComponents,
                                       T const* data, std::size_t dataSize);

    template <typename T>
    tinyxml2::XMLElement* addDataArray(tinyxml2::XMLElement* parent, std::string const& name,
                                       std::size_t inComponents, std::size_t outComponents,
                                       std::vector<T> const& data) {
        return addDataArray(parent, name, inComponents, outComponents, data.data(), data.size());
    }

    std::vector<std::array<double, D>> refNodes_;
    std::vector<unsigned char> appended_;
    tinyxml2::XMLDocument doc_;
};

template <std::size_t D> class VTUPiece {
public:
    VTUPiece(tinyxml2::XMLElement* piece, VTUWriter<D>& writer) : piece_(piece), writer_(writer) {}

    /**
     * @brief Samples FiniteElementFunction and adds point data with name "name" to VTU file.
     */
    void addPointData(std::string const& name, FiniteElementFunction<D> const& function);
    /**
     * @brief Adds cell data with name "name" to VTU file.
     *
     * There is no high order cell data in VTK, therefore pass one value per cell.
     */
    template <typename T>
    void addCellData(std::string const& name, T const* data, std::size_t numElements) {
        auto cdata = piece_->LastChildElement("CellData");
        if (!cdata) {
            cdata = piece_->InsertNewChildElement("CellData");
        }
        writer_.template addDataArray<T>(cdata, name, 1, 1, data, numElements);
    }
    /**
     * @brief Wrapper for addCellData(std::string const&, double const*, std::size_t)
     */
    template <typename T> void addCellData(std::string const& name, std::vector<T> const& data) {
        addCellData(name, data.data(), data.size());
    }

private:
    VTUWriter<D>& writer_;
    tinyxml2::XMLElement* piece_;
};

} // namespace tndm

#endif // VTUWRITER_20200629_H
