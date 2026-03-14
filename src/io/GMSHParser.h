#ifndef GMSHPARSER_20200901_H
#define GMSHPARSER_20200901_H

#include "GMSHLexer.h"
#include "meshParser.h"

#include <array>
#include <optional>
#include <streambuf>
#include <string>
#include <string_view>

namespace tndm {

struct membuf : std::streambuf {
    membuf(char* b, char* e) { this->setg(b, b, e); }
};

class GMSHParser: public meshParser{
private:
    meshBuilder* builder;
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

    GMSHParser(meshBuilder* builder) : builder(builder) {}

    bool parse(std::string& msh);
    bool parse(char* msh, std::size_t len);
    bool parseFile(std::string const& fileName) override;

    std::string_view getErrorMessage() const override { return errorMsg; }
};

}; // namespace tndm

#endif // GMSHPARSER_20200901_H
