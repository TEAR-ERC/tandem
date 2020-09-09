#ifndef GMSHPARSER_20200901_H
#define GMSHPARSER_20200901_H

#include "GMSHLexer.h"

#include <array>
#include <optional>
#include <streambuf>
#include <string>
#include <string_view>

namespace tndm {

struct membuf : std::streambuf {
    membuf(char* b, char* e) { this->setg(b, b, e); }
};

class GMSHMeshBuilder {
public:
    virtual ~GMSHMeshBuilder() {}
    virtual void setNumVertices(std::size_t numVertices) = 0;
    virtual void setVertex(long id, std::array<double, 3> const& x) = 0;
    virtual void setNumElements(std::size_t numElements) = 0;
    virtual void addElement(long type, long tag, long* node, std::size_t numNodes) = 0;
};

class GMSHParser {
private:
    GMSHMeshBuilder* builder;
    GMSHToken curTok;
    GMSHSourceLocation curLoc;
    GMSHLexer lexer;
    std::string errorMsg;

    GMSHToken getNextToken() {
        curLoc = lexer.getSourceLoc();
        return curTok = lexer.getToken();
    }

    template <typename T> T logError(std::string_view msg);
    template <typename T> T logErrorAnnotated(std::string_view msg);
    std::optional<double> getNumber();
    double parseMeshFormat();
    bool parseNodes();
    bool parseElements();
    bool parse_();

public:
    static constexpr std::size_t NumNodes[] = {
        2,  // line
        3,  // triangle,
        4,  // quadrangle,
        4,  // tetrahedron
        8,  // hexahedron
        6,  // prism
        5,  // pyramid
        3,  // P1 line
        6,  // P1 triangle
        9,  // P1 quadrangle
        10, // P1 tetrahedron
        27, // P2 hexahedron
        18, // P2 prism
        14, // P2 pyramid
    };

    GMSHParser(GMSHMeshBuilder* builder) : builder(builder) {}

    bool parse(std::string& msh);
    bool parse(char* msh, std::size_t len);
    bool parseFile(std::string const& fileName);

    std::string_view getErrorMessage() const { return errorMsg; }
};

}; // namespace tndm

#endif // GMSHPARSER_20200901_H
