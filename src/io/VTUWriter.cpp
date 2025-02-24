#include "VTUWriter.h"
#include "Endianness.h"
#include "form/FiniteElementFunction.h"
#include "io/VTUAdapter.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/Scratch.h"

#include <mpi.h>
#include <numeric>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>

#ifdef EXPERIMENTAL_FS
#include <experimental/filesystem>
#else
#include <filesystem>
#endif

#ifdef EXPERIMENTAL_FS
namespace fs = std::experimental::filesystem;
#else
namespace fs = std::filesystem;
#endif

using tinyxml2::XMLAttribute;
using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLPrinter;

namespace tndm {

uint64_t VTKDataArray::make_xml(tinyxml2::XMLElement* parent, uint64_t offset) const {
    auto da = parent->InsertNewChildElement("DataArray");
    da->SetAttribute("type", type_.vtkIdentifier().c_str());
    da->SetAttribute("Name", name_.c_str());
    da->SetAttribute("NumberOfComponents", static_cast<uint64_t>(num_components_));
    da->SetAttribute("format", "appended");
    da->SetAttribute("offset", offset);
    return offset + appended_.size();
}

void VTKDataArray::write_appended(FILE* fp) const {
    std::fwrite(appended_.data(), sizeof(unsigned char), appended_.size(), fp);
}

void VTKDataArray::make_appended(unsigned char const* data, header_t size, bool zlibCompress) {
    unsigned char dummy_data = 0;
    if (data == nullptr) {
        if (size == 0) {
            data = &dummy_data;
        } else {
            throw std::logic_error("VTKDataArray got nullptr but dataSize != 0");
        }
    }

    if (zlibCompress) {
        struct {
            header_t blocks;
            header_t blockSize;
            header_t lastBlockSize;
            header_t compressedBlocksizes;
        } header{1, size, size, 0};
        auto destLen = compressBound(size);
        appended_ = std::vector<unsigned char>(sizeof(header) + destLen);
        unsigned char* app = appended_.data();
        compress(app + sizeof(header), &destLen, data, size);
        header.compressedBlocksizes = destLen;
        memcpy(app, &header, sizeof(header));
        appended_.resize(sizeof(header) + destLen);
    } else {
        appended_ = std::vector<unsigned char>(sizeof(size) + size);
        unsigned char* app = appended_.data();
        memcpy(app, &size, sizeof(size));
        app += sizeof(size);
        memcpy(app, data, size);
    }
}

template <std::size_t D>
VTUWriter<D>::VTUWriter(unsigned degree, bool zlibCompress, MPI_Comm comm)
    : refNodes_(EquidistantNodesFactory<D>(NumberingConvention::VTK)(degree)),
      zlibCompress_(zlibCompress), comm_(comm) {
    MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &group_comm_);
}

template <std::size_t D> VTUWriter<D>::~VTUWriter() { MPI_Comm_free(&group_comm_); }

template <std::size_t D> int32_t VTUPiece<D>::VTKType(bool linear) {
    if constexpr (D == 1u) {
        if (linear) {
            return 3; // VTK_LINE
        }
        return 68; // VTK_LAGRANGE_CURVE
    } else if constexpr (D == 2u) {
        if (linear) {
            return 5; // VTK_TRIANGLE
        }
        return 69; // VTK_LAGRANGE_TRIANGLE
    } else if constexpr (D == 3u) {
        if (linear) {
            return 10; // VTK_TETRA
        }
        return 71; // VTK_LAGRANGE_TETRAHEDRON
    }
    return 0; // VTK_EMPTY_CELL
};

template <std::size_t D> VTUPiece<D>& VTUWriter<D>::addPiece(VTUAdapter<D>& adapter) {
    auto& piece = pieces_.emplace_back(VTUPiece<D>(adapter, refNodes_, zlibCompress_, group_comm_));

    int rank;
    MPI_Comm_rank(comm_, &rank);
    auto partition = std::vector<int32_t>(piece.numElements(), rank);
    piece.addCellData("partition", partition);

    return piece;
}

template <std::size_t D>
VTUPiece<D>::VTUPiece(VTUAdapter<D>& adapter, std::vector<std::array<double, D>> const& refNodes,
                      bool zlibCompress, MPI_Comm group_comm)
    : refNodes_(&refNodes), zlibCompress_(zlibCompress), group_comm_(group_comm) {
    auto pointsPerElement = refNodes_->size();
    num_elements_ = adapter.numElements();
    std::size_t pointDim = adapter.pointDim();

    adapter.setRefNodes(*refNodes_);
    {
        auto vertices = std::vector<double>(num_elements_ * pointsPerElement * PointDim);
        for (std::size_t elNo = 0; elNo < num_elements_; ++elNo) {
            auto firstVertex = &vertices[elNo * pointsPerElement * PointDim];
            auto result = Matrix<double>(firstVertex, pointDim, pointsPerElement);
            adapter.map(elNo, result);
            // Points must be 3d in VTK. Therefore, we set z = 0 for D = 2.
            if (pointDim < PointDim) {
                for (std::ptrdiff_t i = pointsPerElement - 1; i >= 0; --i) {
                    for (std::ptrdiff_t d = PointDim - 1; d >= pointDim; --d) {
                        firstVertex[d + i * PointDim] = 0.0;
                    }
                    for (std::ptrdiff_t d = pointDim - 1; d >= 0; --d) {
                        firstVertex[d + i * PointDim] = firstVertex[d + i * pointDim];
                    }
                }
            }
        }
        auto gathered_points = gather(vertices.data(), vertices.size());
        points_ = VTKDataArray("Points", PointDim, gathered_points, zlibCompress_);
    }

    using size_type = decltype(num_elements_);
    gathered_num_elements_ = 0;
    MPI_Reduce(&num_elements_, &gathered_num_elements_, 1, mpi_type_t<size_type>(), MPI_SUM, 0,
               group_comm_);

    {
        auto connectivity = std::vector<int32_t>(gathered_num_elements_ * pointsPerElement);
        std::iota(connectivity.begin(), connectivity.end(), 0);
        connectivity_ = VTKDataArray("connectivity", 1, connectivity, zlibCompress_);

        auto offsets = std::vector<int32_t>(gathered_num_elements_);
        std::generate(offsets.begin(), offsets.end(),
                      [&pointsPerElement, n = 1]() mutable { return pointsPerElement * n++; });
        offsets_ = VTKDataArray("offsets", 1, offsets, zlibCompress_);

        auto vtkType = VTKType(refNodes_->size() == (D + 1ul));
        auto types = std::vector<uint8_t>(gathered_num_elements_, vtkType);
        types_ = VTKDataArray("types", 1, types, zlibCompress_);
    }
}

template <std::size_t D> void VTUPiece<D>::addPointData(FiniteElementFunction<D> const& function) {
    auto pointsPerElement = refNodes_->size();

    auto E = function.evaluationMatrix(*refNodes_);
    auto data = Managed<Tensor<double, 3u>>(pointsPerElement, function.numElements(),
                                            function.numQuantities());
    auto result = Managed(function.mapResultInfo(pointsPerElement));
    for (std::size_t elNo = 0; elNo < function.numElements(); ++elNo) {
        function.map(elNo, E, result);
        for (std::size_t p = 0; p < function.numQuantities(); ++p) {
            for (std::size_t i = 0; i < pointsPerElement; ++i) {
                data(i, elNo, p) = result(i, p);
            }
        }
    }
    for (std::size_t p = 0; p < function.numQuantities(); ++p) {
        double* data_ptr = nullptr;
        if (function.numElements() > 0) {
            data_ptr = &data(0, 0, p);
        }
        point_data_.emplace_back(VTKDataArray(
            function.name(p), 1, gather(data_ptr, pointsPerElement * function.numElements()),
            zlibCompress_));
    }
}

template <std::size_t D>
void VTUPiece<D>::addJacobianData(FiniteElementFunction<D> const& function,
                                  VTUAdapter<D>& adapter) {
    auto pointsPerElement = refNodes_->size();
    adapter.setRefNodes(*refNodes_);

    auto scratch = Scratch<double>(adapter.scratch_mem_size());
    auto Dxi = function.gradientEvaluationTensor(*refNodes_);
    auto data = Managed<Tensor<double, 3u>>(pointsPerElement, function.numElements(),
                                            D * function.numQuantities());
    auto jInvAtP = Managed(adapter.jacobianResultInfo());
    auto result = Managed(function.gradientResultInfo(pointsPerElement));
    for (std::size_t elNo = 0; elNo < function.numElements(); ++elNo) {
        adapter.jacobianInv(elNo, jInvAtP, scratch);
        function.gradient(elNo, Dxi, jInvAtP, result);
        for (std::size_t d = 0; d < D; ++d) {
            for (std::size_t p = 0; p < function.numQuantities(); ++p) {
                for (std::size_t i = 0; i < pointsPerElement; ++i) {
                    data(i, elNo, d + p * D) = result(i, p, d);
                }
            }
        }
        scratch.reset();
    }
    for (std::size_t d = 0; d < D; ++d) {
        for (std::size_t p = 0; p < function.numQuantities(); ++p) {
            double* data_ptr = nullptr;
            if (function.numElements() > 0) {
                data_ptr = &data(0, 0, d + p * D);
            }
            std::stringstream s;
            s << "d" << function.name(p) << "_d" << d;
            point_data_.emplace_back(VTKDataArray(
                s.str(), 1, gather(data_ptr, pointsPerElement * function.numElements()),
                zlibCompress_));
        }
    }
}

template <std::size_t D> uint64_t VTUPiece<D>::make_xml(XMLElement* parent, uint64_t offset) const {
    auto pointsPerElement = refNodes_->size();
    auto node = parent->InsertNewChildElement("Piece");
    node->SetAttribute("NumberOfPoints",
                       static_cast<uint64_t>(pointsPerElement * gathered_num_elements_));
    node->SetAttribute("NumberOfCells", static_cast<uint64_t>(gathered_num_elements_));

    auto pnode = node->InsertNewChildElement("Points");
    offset = points_.make_xml(pnode, offset);

    auto cnode = node->InsertNewChildElement("Cells");
    offset = connectivity_.make_xml(cnode, offset);
    offset = offsets_.make_xml(cnode, offset);
    offset = types_.make_xml(cnode, offset);

    if (!cell_data_.empty()) {
        auto cdnode = node->InsertNewChildElement("CellData");
        for (auto const& cd : cell_data_) {
            offset = cd.make_xml(cdnode, offset);
        }
    }

    if (!point_data_.empty()) {
        auto pdnode = node->InsertNewChildElement("PointData");
        for (auto const& pd : point_data_) {
            offset = pd.make_xml(pdnode, offset);
        }
    }

    return offset;
}

template <std::size_t D> void VTUPiece<D>::write_appended(FILE* fp) const {
    points_.write_appended(fp);
    connectivity_.write_appended(fp);
    offsets_.write_appended(fp);
    types_.write_appended(fp);
    for (auto const& cd : cell_data_) {
        cd.write_appended(fp);
    }
    for (auto const& pd : point_data_) {
        pd.write_appended(fp);
    }
}

template <std::size_t D>
uint64_t VTUWriter<D>::make_xml(XMLElement* parent, uint64_t offset) const {
    for (auto const& piece : pieces_) {
        offset = piece.make_xml(parent, offset);
    }

    if (!field_data_.empty()) {
        auto fdnode = parent->InsertNewChildElement("FieldData");
        for (auto const& fd : field_data_) {
            offset = fd.make_xml(fdnode, offset);
        }
    }
    return offset;
}

template <std::size_t D> void VTUWriter<D>::write_appended(FILE* fp) const {
    for (auto const& piece : pieces_) {
        piece.write_appended(fp);
    }
    for (auto const& fd : field_data_) {
        fd.write_appended(fp);
    }
}

template <std::size_t D> bool VTUWriter<D>::write(std::string const& baseName) {
    int wrank, grank;
    MPI_Comm_rank(comm_, &wrank);
    MPI_Comm_rank(group_comm_, &grank);

    int is_writer = grank == 0 ? 1 : 0;
    int writer_no;
    int num_writers;
    MPI_Scan(&is_writer, &writer_no, 1, mpi_type_t<int>(), MPI_SUM, comm_);
    MPI_Allreduce(&is_writer, &num_writers, 1, mpi_type_t<int>(), MPI_SUM, comm_);
    writer_no -= 1;

    if (is_writer != 1) {
        return true;
    }

    auto formatName = [](std::string const& baseName, int rk) {
        std::stringstream nameS;
        nameS << baseName << "_" << rk << ".vtu";
        return nameS.str();
    };

    std::string fileName = formatName(baseName, writer_no);
    FILE* fp = std::fopen(fileName.c_str(), "w");
    if (!fp) {
        std::perror("Could not open file for writing");
        return false;
    }

    auto const beginVTU = [&](XMLPrinter& printer, std::string const& type) {
        printer.PushHeader(false, true);
        printer.OpenElement("VTKFile");
        printer.PushAttribute("type", type.c_str());
        printer.PushAttribute("version", "1.0");
        auto headerType = DataType(VTKDataArray::header_t{});
        printer.PushAttribute("header_type", headerType.vtkIdentifier().c_str());
        if (isBigEndian()) {
            printer.PushAttribute("byte_order", "BigEndian");
        } else {
            printer.PushAttribute("byte_order", "LittleEndian");
        }
        if (zlibCompress_) {
            printer.PushAttribute("compressor", "vtkZLibDataCompressor");
        }
    };

    XMLDocument doc;
    auto grid = doc.NewElement("UnstructuredGrid");
    doc.InsertFirstChild(grid);
    make_xml(grid);

    XMLPrinter printer(fp);
    beginVTU(printer, "UnstructuredGrid");
    doc.Print(&printer);
    printer.OpenElement("AppendedData");
    printer.PushAttribute("encoding", "raw");
    printer.PushText("_");
    write_appended(fp);
    std::fputs("\n    </AppendedData>\n</VTKFile>\n", fp);
    std::fclose(fp);

    if (writer_no == 0) {
        auto relative_base = fs::path(baseName).filename().string();
        std::string pvtuName = pvtuFileName(baseName);
        fp = std::fopen(pvtuName.c_str(), "w");
        if (!fp) {
            std::perror("Could not open file for writing");
            return false;
        }
        XMLPrinter printer(fp);
        beginVTU(printer, "PUnstructuredGrid");
        ParallelVTUVisitor pvtu;
        doc.Accept(&pvtu);
        auto& pdoc = pvtu.parallelXML();
        auto grid = pdoc.FirstChildElement("PUnstructuredGrid");
        if (grid) {
            grid->SetAttribute("GhostLevel", 0);
            int commSize;
            MPI_Comm_size(comm_, &commSize);
            for (int rk = 0; rk < num_writers; ++rk) {
                auto piece = grid->InsertNewChildElement("Piece");
                piece->SetAttribute("Source", formatName(relative_base, rk).c_str());
            }
        }
        pdoc.Print(&printer);
        printer.CloseElement(); // VTKFile
        std::fclose(fp);
    }

    return true;
}

bool ParallelVTUVisitor::VisitEnter(XMLElement const& element, XMLAttribute const* attribute) {
    if (strcmp(element.Name(), "Piece") != 0) {
        auto newName = "P" + std::string(element.Name());
        auto clone = pdoc_.NewElement(newName.c_str());
        while (attribute) {
            if (strcmp(attribute->Name(), "format") != 0 &&
                strcmp(attribute->Name(), "offset") != 0) {
                clone->SetAttribute(attribute->Name(), attribute->Value());
            }
            attribute = attribute->Next();
        }
        subtree_ = subtree_->InsertEndChild(clone);
    }
    return true;
}

bool ParallelVTUVisitor::VisitExit(tinyxml2::XMLElement const& element) {
    if (strcmp(element.Name(), "Piece") != 0) {
        subtree_ = subtree_->Parent();
    }
    return true;
}

template class VTUPiece<1u>;
template class VTUPiece<2u>;
template class VTUPiece<3u>;
template class VTUWriter<1u>;
template class VTUWriter<2u>;
template class VTUWriter<3u>;

} // namespace tndm
