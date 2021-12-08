#ifndef VTUWRITER_20200629_H
#define VTUWRITER_20200629_H

#include "DataType.h"
#include "basis/Equidistant.h"
#include "basis/NumberingConvention.h"
#include "parallel/CommPattern.h"
#include "parallel/MPITraits.h"

#include <mpi.h>
#include <stdexcept>
#include <tinyxml2.h>
#include <zlib.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace tndm {

template <std::size_t D> class FiniteElementFunction;
template <std::size_t D> class VTUAdapter;
template <std::size_t D> class VTUPiece;

class VTKDataArray {
public:
    using header_t = uint64_t;

    VTKDataArray() {}

    template <typename T>
    VTKDataArray(std::string const& name, std::size_t num_components, T const* data,
                 std::size_t dataSize, bool zlibCompress)
        : type_(DataType(T{})), name_(name), num_components_(num_components) {
        unsigned const char* d = reinterpret_cast<unsigned const char*>(data);
        header_t size = dataSize * sizeof(T);
        make_appended(d, size, zlibCompress);
    }

    template <typename T>
    VTKDataArray(std::string const& name, std::size_t num_components, std::vector<T> const& data,
                 bool zlibCompress)
        : type_(DataType(T{})), name_(name), num_components_(num_components) {
        unsigned const char* d = reinterpret_cast<unsigned const char*>(data.data());
        header_t size = data.size() * sizeof(T);
        make_appended(d, size, zlibCompress);
    }

    uint64_t make_xml(tinyxml2::XMLElement* parent, uint64_t offset) const;
    void write_appended(FILE* fp) const;

private:
    void make_appended(unsigned char const* data, header_t size, bool zlibCompress);

    DataType type_;
    std::string name_;
    uint64_t num_components_;
    std::vector<unsigned char> appended_;
};

template <std::size_t D> class VTUWriter {
public:
    VTUWriter(unsigned degree = 1u, bool zlibCompress = true, MPI_Comm comm = MPI_COMM_WORLD);
    ~VTUWriter();

    VTUPiece<D>& addPiece(VTUAdapter<D>& adapter);

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
        field_data_.emplace_back(VTKDataArray(name, 1, data, numElements, zlibCompress_));
    }
    /**
     * @brief Wrapper for addFieldData(std::string const&, double const*, std::size_t)
     */
    template <typename T> void addFieldData(std::string const& name, std::vector<T> const& data) {
        field_data_.emplace_back(VTKDataArray(name, 1, data, zlibCompress_));
    }

    uint64_t make_xml(tinyxml2::XMLElement* parent, uint64_t offset = 0) const;
    void write_appended(FILE* fp) const;

private:
    std::vector<VTKDataArray> field_data_;
    std::vector<VTUPiece<D>> pieces_;

    std::vector<std::array<double, D>> refNodes_;
    bool zlibCompress_;
    MPI_Comm comm_, group_comm_;
};

template <std::size_t D> class VTUPiece {
public:
    static constexpr std::size_t PointDim = 3u;
    static int32_t VTKType(bool linear);

    VTUPiece(VTUAdapter<D>& adapter, std::vector<std::array<double, D>> const& refNodes,
             bool zlibCompress, MPI_Comm group_comm);

    inline auto numElements() const { return num_elements_; }

    /**
     * @brief Samples FiniteElementFunction and adds point data to VTU file.
     */
    void addPointData(FiniteElementFunction<D> const& function);
    /**
     * @brief Samples Jacobian of FiniteElementFunction and adds point data to VTU file.
     */
    void addJacobianData(FiniteElementFunction<D> const& function, VTUAdapter<D>& adapter);
    /**
     * @brief Adds cell data with name "name" to VTU file.
     *
     * There is no high order cell data in VTK, therefore pass one value per cell.
     */
    template <typename T>
    void addCellData(std::string const& name, T const* data, std::size_t numElements) {
        cell_data_.emplace_back(VTKDataArray(name, 1, gather(data, numElements), zlibCompress_));
    }
    /**
     * @brief Wrapper for addCellData(std::string const&, double const*, std::size_t)
     */
    template <typename T> void addCellData(std::string const& name, std::vector<T> const& data) {
        addCellData(name, data.data(), data.size());
    }

    uint64_t make_xml(tinyxml2::XMLElement* parent, uint64_t offset) const;
    void write_appended(FILE* fp) const;

private:
    template <typename T> auto gather(T const* data, std::size_t numElements) -> std::vector<T> {
        return GatherV(static_cast<int>(numElements), 0, group_comm_).exchange(data);
    }

    std::vector<std::array<double, D>> const* refNodes_;
    bool zlibCompress_;
    MPI_Comm group_comm_;

    std::size_t num_elements_, gathered_num_elements_;
    VTKDataArray points_, connectivity_, offsets_, types_;
    std::vector<VTKDataArray> point_data_;
    std::vector<VTKDataArray> cell_data_;
};

class ParallelVTUVisitor : public tinyxml2::XMLVisitor {
public:
    ParallelVTUVisitor() : subtree_(&pdoc_) {}

    bool VisitEnter(tinyxml2::XMLElement const& element,
                    tinyxml2::XMLAttribute const* attribute) override;
    bool VisitExit(tinyxml2::XMLElement const& element) override;

    auto parallelXML() -> tinyxml2::XMLDocument& { return pdoc_; }

private:
    tinyxml2::XMLDocument pdoc_;
    tinyxml2::XMLNode* subtree_;
};

} // namespace tndm

#endif // VTUWRITER_20200629_H
