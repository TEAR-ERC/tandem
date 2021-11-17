#include "GMSHParser.h"
#include "io/GMSHLexer.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace tndm {

template <typename T> T GMSHParser::logErrorAnnotated(std::string_view msg) {
    std::stringstream ss;
    ss << "GMSH parser error in line " << curLoc.line << " in column " << curLoc.col << ":\n";
    ss << '\t' << msg << '\n';
    errorMsg += ss.str();
    return {};
}

template <typename T> T GMSHParser::logError(std::string_view msg) {
    errorMsg += "GMSH parser error:\n\t";
    errorMsg += msg;
    errorMsg += '\n';
    return {};
}

std::optional<double> GMSHParser::getNumber() {
    if (curTok == GMSHToken::integer) {
        return {lexer.getInteger()};
    } else if (curTok == GMSHToken::real) {
        return {lexer.getReal()};
    }
    return std::nullopt;
}

bool GMSHParser::parse(char* msh, std::size_t len) {
    membuf buf(msh, msh + len);
    std::istream in(&buf);
    lexer.setIStream(&in);
    return parse_();
}

bool GMSHParser::parse(std::string& msh) {
    membuf buf(msh.data(), msh.data() + msh.size());
    std::istream in(&buf);
    lexer.setIStream(&in);
    return parse_();
}

bool GMSHParser::parseFile(std::string const& fileName) {
    std::ifstream in(fileName);
    if (!in.is_open()) {
        return logError<bool>("Unable to open MSH file");
    }
    lexer.setIStream(&in);
    return parse_();
}

bool GMSHParser::parse_() {
    errorMsg.clear();
    getNextToken();

    double version = parseMeshFormat();
    if (version < 2.0 || version >= 3.0) {
        char buf[128];
        sprintf(buf, "Unsupported MSH version %.1lf", version);
        return logError<bool>(buf);
    }

    bool hasNodes = false;
    bool hasElements = false;

    while (curTok != GMSHToken::eof) {
        switch (curTok) {
        case GMSHToken::nodes:
            parseNodes();
            hasNodes = true;
            break;
        case GMSHToken::elements:
            parseElements();
            hasElements = true;
            break;
        default:
            getNextToken();
            break;
        }
    }

    return hasNodes && hasElements;
}

double GMSHParser::parseMeshFormat() {
    if (curTok != GMSHToken::mesh_format) {
        return logErrorAnnotated<double>("Expected $MeshFormat");
    }
    getNextToken();
    auto version = getNumber();
    if (!version) {
        return logErrorAnnotated<double>("Expected version number");
    }
    getNextToken();
    if (curTok != GMSHToken::integer || lexer.getInteger() != 0) {
        return logErrorAnnotated<double>("Expected 0");
    }
    getNextToken(); // skip data-size
    getNextToken();
    if (curTok != GMSHToken::end_mesh_format) {
        return logErrorAnnotated<double>("Expected $EndMeshFormat");
    }
    getNextToken();
    return *version;
}

bool GMSHParser::parseNodes() {
    getNextToken();
    if (curTok != GMSHToken::integer || lexer.getInteger() < 0) {
        return logErrorAnnotated<bool>("Expected non-zero integer");
    }
    std::size_t numVertices = lexer.getInteger();
    builder->setNumVertices(numVertices);

    for (std::size_t i = 0; i < numVertices; ++i) {
        getNextToken();
        if (curTok != GMSHToken::integer || lexer.getInteger() < 1 ||
            lexer.getInteger() > numVertices) {
            char buf[128];
            sprintf(buf, "Expected node-tag with 1 <= node-tag <= %zu", numVertices);
            return logErrorAnnotated<bool>(buf);
        }
        std::size_t id = lexer.getInteger() - 1;

        std::array<double, 3> x;
        for (std::size_t i = 0; i < 3; ++i) {
            getNextToken();
            auto coord = getNumber();
            if (!coord) {
                return logErrorAnnotated<bool>("Expected coordinate");
            }
            x[i] = *coord;
        }
        builder->setVertex(id, x);
    }
    getNextToken();
    if (curTok != GMSHToken::end_nodes) {
        return logErrorAnnotated<bool>("Expected $EndNodes");
    }
    getNextToken();
    return true;
}

bool GMSHParser::parseElements() {
    getNextToken();
    if (curTok != GMSHToken::integer || lexer.getInteger() < 0) {
        return logErrorAnnotated<bool>("Expected positive integer");
    }
    auto numElements = lexer.getInteger();

    constexpr std::size_t MaxElementType = sizeof(NumNodes) / sizeof(std::size_t);
    constexpr std::size_t MaxNodes = *std::max_element(NumNodes, NumNodes + MaxElementType);
    long tag = -1;
    std::array<long, MaxNodes> nodes;

    builder->setNumElements(numElements);

    for (std::size_t i = 0; i < numElements; ++i) {
        getNextToken();
        if (curTok != GMSHToken::integer) {
            return logErrorAnnotated<bool>("Expected element-tag");
        }

        getNextToken();
        if (curTok != GMSHToken::integer || lexer.getInteger() < 1 ||
            lexer.getInteger() > MaxElementType) {
            char buf[128];
            sprintf(buf, "Expected element-type with 1 <= element-type <= %zu", MaxElementType);
            return logErrorAnnotated<bool>(buf);
        }
        long type = lexer.getInteger();

        getNextToken();
        if (curTok != GMSHToken::integer || lexer.getInteger() < 0) {
            return logErrorAnnotated<bool>("Expected number of tags");
        }
        long numTags = lexer.getInteger();
        for (long i = 0; i < numTags; ++i) {
            getNextToken();
            if (curTok != GMSHToken::integer) {
                return logErrorAnnotated<bool>("Expected tag (integer)");
            }
            if (i == 0) {
                tag = lexer.getInteger();
            }
        }

        for (std::size_t i = 0; i < NumNodes[type - 1]; ++i) {
            getNextToken();
            if (curTok != GMSHToken::integer || lexer.getInteger() < 1) {
                char buf[128];
                sprintf(buf, "Expected node number > 0 (%zu/%zu for type %li)", i + 1,
                        NumNodes[type - 1], type);
                return logErrorAnnotated<bool>(buf);
            }
            nodes[i] = lexer.getInteger() - 1;
        }

        builder->addElement(type, tag, nodes.data(), NumNodes[type - 1]);
    }
    getNextToken();
    if (curTok != GMSHToken::end_elements) {
        return logErrorAnnotated<bool>("Expected $EndElements");
    }
    getNextToken();

    return true;
}

} // namespace tndm
