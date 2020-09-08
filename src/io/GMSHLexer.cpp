#include "GMSHLexer.h"
#include "util/Hash.h"

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace tndm {

void GMSHLexer::advance() {
    in->get(lastChar);

    if (lastChar == '\n' || lastChar == '\r') {
        ++loc.line;
        loc.col = 1;
    } else {
        ++loc.col;
    }
}

GMSHToken GMSHLexer::getToken() {
    if (in == nullptr) {
        return GMSHToken::eof;
    }
    in->peek();
    if (!in->good()) {
        return GMSHToken::eof;
    }

    while (isspace(lastChar)) {
        advance();
    }

    if (lastChar == '$') {
        advance();
        auto hash = fnv1a0();
        while (isalpha(lastChar)) {
            hash = fnv1a_step(hash, lastChar);
            advance();
        }
        GMSHToken token = GMSHToken::unknown_section;
        switch (hash) {
        case "MeshFormat"_fnv1a:
            token = GMSHToken::mesh_format;
            break;
        case "EndMeshFormat"_fnv1a:
            token = GMSHToken::end_mesh_format;
            break;
        case "Nodes"_fnv1a:
            token = GMSHToken::nodes;
            break;
        case "EndNodes"_fnv1a:
            token = GMSHToken::end_nodes;
            break;
        case "Elements"_fnv1a:
            token = GMSHToken::elements;
            break;
        case "EndElements"_fnv1a:
            token = GMSHToken::end_elements;
            break;
        default:
            break;
        }
        return token;
    }

    auto mustbereal = [](char c) { return c == '.' || c == 'e' || c == 'E'; };
    auto isnumber = [&mustbereal](char c) {
        return isdigit(c) || c == '+' || c == '-' || mustbereal(c);
    };

    if (isnumber(lastChar)) {
        char buf[MaxNumberLength + 1];
        int pos = 0;
        bool isreal = false;
        do {
            buf[pos++] = lastChar;
            isreal = isreal || mustbereal(lastChar);
            advance();
        } while (isnumber(lastChar) && pos < MaxNumberLength);
        buf[pos] = 0;
        if (pos == MaxNumberLength) {
            throw std::runtime_error("Too large number encountered in GMSHLexer: " +
                                     std::string(buf));
        }
        if (isreal) {
            real = std::strtod(buf, 0);
            return GMSHToken::real;
        }
        integer = std::strtol(buf, 0, 10);
        return GMSHToken::integer;
    }

    if (lastChar == '"') {
        do {
            advance();
        } while (lastChar != '"');
        advance();
        return GMSHToken::string;
    }
    return GMSHToken::unknown_token;
}

} // namespace tndm
