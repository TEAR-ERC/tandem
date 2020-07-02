#include "VTUWriter.h"
#include "DataType.h"
#include "Endianness.h"

#include <functional>
#include <mpi.h>
#include <numeric>
#include <tinyxml2.h>

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
    ;
    piece->SetAttribute("NumberOfPoints", pointsPerElement * cl.numElements());
    piece->SetAttribute("NumberOfCells", cl.numElements());
    {
        auto points = piece->InsertNewChildElement("Points");
        auto E = cl.evaluateBasisAt(refNodes_);
        auto vertices = std::vector<double>(cl.numElements() * pointsPerElement * D);
        for (std::size_t elNo = 0; elNo < cl.numElements(); ++elNo) {
            auto result =
                Matrix<double>(&vertices[elNo * pointsPerElement * D], D, pointsPerElement);
            cl.map(elNo, E, result);
        }
        addDataArray(points, "Points", D, PointDim, vertices);
    }

    auto cells = piece->InsertNewChildElement("Cells");
    {
        auto connectivity = std::vector<int32_t>(cl.numElements() * pointsPerElement);
        std::iota(connectivity.begin(), connectivity.end(), 0);
        addDataArray(cells, "connectivity", 1, 1, connectivity);

        auto offsets = std::vector<int32_t>(cl.numElements());
        std::generate(offsets.begin(), offsets.end(),
                      [&pointsPerElement, n = 1]() mutable { return pointsPerElement * n++; });
        addDataArray(cells, "offsets", 1, 1, offsets);

        auto vtkType = VTKType(refNodes_.size() == (D + 1ul));
        auto types = std::vector<int32_t>(cl.numElements(), vtkType);
        addDataArray(cells, "types", 1, 1, types);
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
    writer_.addDataArray(pdata, name, 1, 1, data);
}

template <std::size_t D>
template <typename T>
XMLElement* VTUWriter<D>::addDataArray(XMLElement* parent, std::string const& name,
                                       std::size_t inComponents, std::size_t outComponents,
                                       T const* data, std::size_t dataSize) {
    auto da = parent->InsertNewChildElement("DataArray");
    auto dataType = DataType(T{});
    da->SetAttribute("type", dataType.vtkIdentifier().c_str());
    da->SetAttribute("Name", name.c_str());
    da->SetAttribute("NumberOfComponents", outComponents);
    da->SetAttribute("format", "appended");

    auto offset = appended_.size();
    da->SetAttribute("offset", offset);

    assert(dataSize % inComponents == 0);
    auto num = dataSize / inComponents;
    header_t size = num * outComponents * sizeof(T);

    appended_.resize(appended_.size() + sizeof(size) + size);
    unsigned char* app = appended_.data() + offset;
    memcpy(app, &size, sizeof(size));
    app += sizeof(size);
    if (inComponents == outComponents) {
        memcpy(app, data, dataSize * sizeof(T));
    } else if (outComponents > inComponents) {
        for (std::size_t i = 0; i < num; ++i) {
            memcpy(app, &data[i * inComponents], inComponents * sizeof(T));
            memset(app + inComponents * sizeof(T), 0, (outComponents - inComponents) * sizeof(T));
            app += outComponents * sizeof(T);
        }
    } else {
        assert(false);
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
