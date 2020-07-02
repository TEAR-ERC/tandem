#include "VTUWriter.h"
#include "DataType.h"
#include "Endianness.h"

#include <functional>
#include <mpi.h>
#include <numeric>
#include <tinyxml2.h>
#include <zlib.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>

using tinyxml2::XMLDocument;
using tinyxml2::XMLElement;
using tinyxml2::XMLPrinter;

namespace tndm {

template <std::size_t D> int32_t VTUWriter<D>::VTKType(bool linear) {
    if constexpr (D == 2u) {
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

template <std::size_t D> VTUPiece<D> VTUWriter<D>::addPiece(Curvilinear<D>& cl) {
    auto pointsPerElement = refNodes_.size();

    auto piece = doc_.RootElement()->InsertNewChildElement("Piece");
    piece->SetAttribute("NumberOfPoints", pointsPerElement * cl.numElements());
    piece->SetAttribute("NumberOfCells", cl.numElements());
    {
        auto points = piece->InsertNewChildElement("Points");
        auto E = cl.evaluateBasisAt(refNodes_);
        auto vertices = std::vector<double>(cl.numElements() * pointsPerElement * PointDim);
        for (std::size_t elNo = 0; elNo < cl.numElements(); ++elNo) {
            auto firstVertex = &vertices[elNo * pointsPerElement * PointDim];
            auto result = Matrix<double>(firstVertex, D, pointsPerElement);
            cl.map(elNo, E, result);
            // Points must be 3d in VTK. Therefore, we set z = 0 for D = 2.
            if constexpr (D < PointDim) {
                for (std::ptrdiff_t i = pointsPerElement - 1; i >= 0; --i) {
                    for (std::ptrdiff_t d = PointDim - 1; d >= D; --d) {
                        firstVertex[d + i * PointDim] = 0.0;
                    }
                    for (std::ptrdiff_t d = D - 1; d >= 0; --d) {
                        firstVertex[d + i * PointDim] = firstVertex[d + i * D];
                    }
                }
            }
        }
        addDataArray(points, "Points", PointDim, vertices);
    }

    auto cells = piece->InsertNewChildElement("Cells");
    {
        auto connectivity = std::vector<int32_t>(cl.numElements() * pointsPerElement);
        std::iota(connectivity.begin(), connectivity.end(), 0);
        addDataArray(cells, "connectivity", 1, connectivity);

        auto offsets = std::vector<int32_t>(cl.numElements());
        std::generate(offsets.begin(), offsets.end(),
                      [&pointsPerElement, n = 1]() mutable { return pointsPerElement * n++; });
        addDataArray(cells, "offsets", 1, offsets);

        auto vtkType = VTKType(refNodes_.size() == (D + 1ul));
        auto types = std::vector<int32_t>(cl.numElements(), vtkType);
        addDataArray(cells, "types", 1, types);
    }

    return VTUPiece<D>(piece, *this);
}

template <std::size_t D>
void VTUPiece<D>::addPointData(std::string const& name, FiniteElementFunction<D> const& function) {
    auto pointsPerElement = writer_.refNodes().size();

    // TODO: add support for more than one quantity
    assert(function.numQuantities() == 1);

    XMLElement* pdata = piece_->LastChildElement("PointData");
    if (!pdata) {
        pdata = piece_->InsertNewChildElement("PointData");
    }

    auto E = function.evaluationMatrix(writer_.refNodes());
    auto data = std::vector<double>(function.numElements() * pointsPerElement);
    for (std::size_t elNo = 0; elNo < function.numElements(); ++elNo) {
        auto result = Matrix<double>(&data[elNo * pointsPerElement], pointsPerElement, 1);
        function.map(elNo, E, result);
    }
    writer_.addDataArray(pdata, name, 1, data);
}

template <std::size_t D>
template <typename T>
XMLElement* VTUWriter<D>::addDataArray(XMLElement* parent, std::string const& name,
                                       std::size_t numComponents, T const* data,
                                       std::size_t dataSize) {
    auto da = parent->InsertNewChildElement("DataArray");
    auto dataType = DataType(T{});
    da->SetAttribute("type", dataType.vtkIdentifier().c_str());
    da->SetAttribute("Name", name.c_str());
    da->SetAttribute("NumberOfComponents", numComponents);
    da->SetAttribute("format", "appended");

    auto offset = appended_.size();
    da->SetAttribute("offset", offset);

    header_t size = dataSize * sizeof(T);

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
        compress(app + sizeof(header), &destLen, reinterpret_cast<unsigned char const*>(data), size);
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

template <std::size_t D> bool VTUWriter<D>::write(std::string const& baseName) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::stringstream nameS;
    nameS << baseName << "-" << rank << ".vtu";
    std::string fileName = nameS.str();

    FILE* fp = std::fopen(fileName.c_str(), "w");
    if (!fp) {
        std::perror("Could not open file for writing");
        return false;
    }

    XMLPrinter printer(fp);
    printer.PushHeader(false, true);
    printer.OpenElement("VTKFile");
    printer.PushAttribute("type", "UnstructuredGrid");
    printer.PushAttribute("version", "1.0");
    auto headerType = DataType(header_t{});
    printer.PushAttribute("header_type", headerType.vtkIdentifier().c_str());
    if (isBigEndian()) {
        printer.PushAttribute("byte_order", "BigEndian");
    } else {
        printer.PushAttribute("byte_order", "LittleEndian");
    }
    if (zlibCompress_) {
        printer.PushAttribute("compressor", "vtkZLibDataCompressor");
    }
    doc_.Print(&printer);
    printer.OpenElement("AppendedData");
    printer.PushAttribute("encoding", "raw");
    printer.PushText("_");
    std::fwrite(appended_.data(), sizeof(unsigned char), appended_.size(), fp);
    std::fputs("\n    </AppendedData>\n</VTKFile>\n", fp);
    std::fclose(fp);

    return true;
}

template class VTUPiece<2u>;
template class VTUPiece<3u>;
template class VTUWriter<2u>;
template class VTUWriter<3u>;

} // namespace tndm
