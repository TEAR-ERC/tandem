#ifndef VTUWRITER_20200629_H
#define VTUWRITER_20200629_H

#include "DataType.h"
#include "basis/Equidistant.h"

#include <mpi.h>
#include <stdexcept>
#include <tinyxml2.h>
#include <zlib.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace tndm {

template <std::size_t D> class FiniteElementFunction;
template <std::size_t D> class VTUAdapter;
template <std::size_t D> class VTUPiece;

template <std::size_t D> class VTUWriter {
public:
    static constexpr std::size_t PointDim = 3u;
    using header_t = uint64_t;

    static int32_t VTKType(bool linear);

    VTUWriter(unsigned degree = 1u, bool zlibCompress = true, MPI_Comm comm = MPI_COMM_WORLD)
        : refNodes_(EquidistantNodesFactory<D>()(degree)), zlibCompress_(zlibCompress),
          comm_(comm) {
        auto grid = doc_.NewElement("UnstructuredGrid");
        doc_.InsertFirstChild(grid);
    }

    VTUPiece<D> addPiece(VTUAdapter<D>& adapter);

    /**
     * @brief Write VTU to disk.
     *
     * @param baseName File name without extension
     *
     * @return True if write was successful.
     */
    bool write(std::string const& baseName);
    std::string pvtuFileName(std::string const& baseName) const { return baseName + ".pvtu"; }

    std::vector<std::array<double, D>> const& refNodes() const { return refNodes_; }

    /**
     * @brief Adds field data with name "name" to VTU file.
     */
    template <typename T>
    void addFieldData(std::string const& name, T const* data, std::size_t numElements) {
        auto cdata = doc_.RootElement()->LastChildElement("FieldData");
        if (!cdata) {
            cdata = doc_.RootElement()->InsertNewChildElement("FieldData");
        }
        auto da = addDataArray<T>(cdata, name, 1, data, numElements);
        da->SetAttribute("NumberOfTuples", 1);
    }
    /**
     * @brief Wrapper for addFieldData(std::string const&, double const*, std::size_t)
     */
    template <typename T> void addFieldData(std::string const& name, std::vector<T> const& data) {
        addFieldData(name, data.data(), data.size());
    }

private:
    friend class VTUPiece<D>;

    template <typename T>
    tinyxml2::XMLElement* addDataArray(tinyxml2::XMLElement* parent, std::string const& name,
                                       std::size_t numComponents, T const* data,
                                       std::size_t dataSize) {
        auto da = parent->InsertNewChildElement("DataArray");
        auto dataType = DataType(T{});
        da->SetAttribute("type", dataType.vtkIdentifier().c_str());
        da->SetAttribute("Name", name.c_str());
        da->SetAttribute("NumberOfComponents", static_cast<uint64_t>(numComponents));
        da->SetAttribute("format", "appended");

        auto offset = appended_.size();
        da->SetAttribute("offset", static_cast<uint64_t>(offset));

        header_t size = dataSize * sizeof(T);

        T dummy_data = 0;
        if (data == nullptr) {
            if (dataSize == 0) {
                data = &dummy_data;
            } else {
                throw std::logic_error("addDataArray got nullptr but dataSize != 0");
            }
        }

        if (zlibCompress_) {
            struct {
                header_t blocks;
                header_t blockSize;
                header_t lastBlockSize;
                header_t compressedBlocksizes;
            } header{1, size, size, 0};
            auto destLen = compressBound(size);
            appended_.resize(offset + sizeof(header) + destLen);
            unsigned char* app = appended_.data() + offset;
            compress(app + sizeof(header), &destLen, reinterpret_cast<unsigned char const*>(data),
                     size);
            header.compressedBlocksizes = destLen;
            memcpy(app, &header, sizeof(header));
            appended_.resize(offset + sizeof(header) + destLen);
        } else {
            appended_.resize(appended_.size() + sizeof(size) + size);
            unsigned char* app = appended_.data() + offset;
            memcpy(app, &size, sizeof(size));
            app += sizeof(size);
            memcpy(app, data, size);
        }
        return da;
    }

    template <typename T>
    tinyxml2::XMLElement* addDataArray(tinyxml2::XMLElement* parent, std::string const& name,
                                       std::size_t numComponents, std::vector<T> const& data) {
        return addDataArray(parent, name, numComponents, data.data(), data.size());
    }

    std::vector<std::array<double, D>> refNodes_;
    bool zlibCompress_;
    MPI_Comm comm_;
    std::vector<unsigned char> appended_;
    tinyxml2::XMLDocument doc_;
};

template <std::size_t D> class VTUPiece {
public:
    VTUPiece(tinyxml2::XMLElement* piece, VTUWriter<D>& writer) : piece_(piece), writer_(writer) {}

    /**
     * @brief Samples FiniteElementFunction and adds point data to VTU file.
     */
    void addPointData(FiniteElementFunction<D> const& function);
    /**
     * @brief Samples Jacobian of FiniteElementFunction and adds point data to VTU file.
     */
    void addJacobianData(FiniteElementFunction<D> const& function, VTUAdapter<D> const& adapter);
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
        writer_.template addDataArray<T>(cdata, name, 1, data, numElements);
    }
    /**
     * @brief Wrapper for addCellData(std::string const&, double const*, std::size_t)
     */
    template <typename T> void addCellData(std::string const& name, std::vector<T> const& data) {
        addCellData(name, data.data(), data.size());
    }

private:
    tinyxml2::XMLElement* getPointDataRoot();

    VTUWriter<D>& writer_;
    tinyxml2::XMLElement* piece_;
};

class ParallelVTUVisitor : public tinyxml2::XMLVisitor {
public:
    ParallelVTUVisitor() : subtree_(&pdoc_) {}

    bool VisitEnter(tinyxml2::XMLElement const& element,
                    tinyxml2::XMLAttribute const* attribute) override;
    bool VisitExit(tinyxml2::XMLElement const& element) override;

    auto& parallelXML() { return pdoc_; }

private:
    tinyxml2::XMLDocument pdoc_;
    tinyxml2::XMLNode* subtree_;
};

} // namespace tndm

#endif // VTUWRITER_20200629_H
