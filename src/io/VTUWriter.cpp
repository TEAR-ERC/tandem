#include "VTUWriter.h"
#include "Endianness.h"

#include <functional>
#include <mpi.h>
#include <numeric>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>

using tinyxml2::XMLAttribute;
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

template <std::size_t D>
VTUPiece<D> VTUWriter<D>::addPiece(Curvilinear<D>& cl, Range<std::size_t> elementRange) {
    assert(elementRange.from <= cl.numElements() && elementRange.to <= cl.numElements());

    auto pointsPerElement = refNodes_.size();

    auto piece = doc_.RootElement()->InsertNewChildElement("Piece");
    piece->SetAttribute("NumberOfPoints", pointsPerElement * elementRange.size());
    piece->SetAttribute("NumberOfCells", elementRange.size());
    {
        auto points = piece->InsertNewChildElement("Points");
        auto E = cl.evaluateBasisAt(refNodes_);
        auto vertices = std::vector<double>(elementRange.size() * pointsPerElement * PointDim);
        for (std::size_t elNo = elementRange.from; elNo < elementRange.to; ++elNo) {
            auto firstVertex = &vertices[(elNo - elementRange.from) * pointsPerElement * PointDim];
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
        auto connectivity = std::vector<int32_t>(elementRange.size() * pointsPerElement);
        std::iota(connectivity.begin(), connectivity.end(), 0);
        addDataArray(cells, "connectivity", 1, connectivity);

        auto offsets = std::vector<int32_t>(elementRange.size());
        std::generate(offsets.begin(), offsets.end(),
                      [&pointsPerElement, n = 1]() mutable { return pointsPerElement * n++; });
        addDataArray(cells, "offsets", 1, offsets);

        auto vtkType = VTKType(refNodes_.size() == (D + 1ul));
        auto types = std::vector<uint8_t>(elementRange.size(), vtkType);
        addDataArray(cells, "types", 1, types);
    }

    auto vtupiece = VTUPiece<D>(piece, *this);
    int rank;
    MPI_Comm_rank(comm_, &rank);
    auto partition = std::vector<int32_t>(elementRange.size(), rank);
    vtupiece.addCellData("partition", partition);
    return vtupiece;
}

template <std::size_t D>
void VTUPiece<D>::addPointData(std::string const& name, FiniteElementFunction<D> const& function) {
    auto pointsPerElement = writer_.refNodes().size();

    XMLElement* pdata = piece_->LastChildElement("PointData");
    if (!pdata) {
        pdata = piece_->InsertNewChildElement("PointData");
    }

    auto E = function.evaluationMatrix(writer_.refNodes());
    auto data = Managed<Tensor<double, 3u>>(pointsPerElement, function.numElements(),
                                            function.numQuantities());
    auto result = Managed<Matrix<double>>(pointsPerElement, function.numQuantities());
    for (std::size_t elNo = 0; elNo < function.numElements(); ++elNo) {
        function.map(elNo, E, result);
        for (std::size_t p = 0; p < function.numQuantities(); ++p) {
            for (std::size_t i = 0; i < pointsPerElement; ++i) {
                data(i, elNo, p) = result(i, p);
            }
        }
    }
    for (std::size_t p = 0; p < function.numQuantities(); ++p) {
        std::string fname;
        if (function.numQuantities() == 1) {
            fname = name;
        } else {
            std::stringstream ss;
            ss << name << p;
            fname = ss.str();
        }
        writer_.addDataArray(pdata, fname, 1, &data(0, 0, p),
                             pointsPerElement * function.numElements());
    }
}

template <std::size_t D> bool VTUWriter<D>::write(std::string const& baseName) {
    int rank;
    MPI_Comm_rank(comm_, &rank);
    auto formatName = [&baseName](int rk) {
        std::stringstream nameS;
        nameS << baseName << "-" << rk << ".vtu";
        return nameS.str();
    };

    std::string fileName = formatName(rank);
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
    };

    XMLPrinter printer(fp);
    beginVTU(printer, "UnstructuredGrid");
    doc_.Print(&printer);
    printer.OpenElement("AppendedData");
    printer.PushAttribute("encoding", "raw");
    printer.PushText("_");
    std::fwrite(appended_.data(), sizeof(unsigned char), appended_.size(), fp);
    std::fputs("\n    </AppendedData>\n</VTKFile>\n", fp);
    std::fclose(fp);

    if (rank == 0) {
        std::string pvtuName = baseName + ".pvtu";
        fp = std::fopen(pvtuName.c_str(), "w");
        if (!fp) {
            std::perror("Could not open file for writing");
            return false;
        }
        XMLPrinter printer(fp);
        beginVTU(printer, "PUnstructuredGrid");
        ParallelVTUVisitor pvtu;
        doc_.Accept(&pvtu);
        auto& pdoc = pvtu.parallelXML();
        auto grid = pdoc.FirstChildElement("PUnstructuredGrid");
        if (grid) {
            grid->SetAttribute("GhostLevel", 0);
            int commSize;
            MPI_Comm_size(comm_, &commSize);
            for (int rk = 0; rk < commSize; ++rk) {
                auto piece = grid->InsertNewChildElement("Piece");
                piece->SetAttribute("Source", formatName(rk).c_str());
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

template class VTUPiece<2u>;
template class VTUPiece<3u>;
template class VTUWriter<2u>;
template class VTUWriter<3u>;

} // namespace tndm
