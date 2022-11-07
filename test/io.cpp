#include "io/GMSHLexer.h"
#include "io/GMSHParser.h"

#include "doctest.h"

#include <array>
#include <iostream>
#include <string_view>

using namespace tndm;

class MyTestBuilder : public GMSHMeshBuilder {
private:
    std::size_t elNo = 0;

public:
    void setNumVertices(std::size_t numVertices) override { CHECK(numVertices == 6u); }
    void setVertex(long id, std::array<double, 3> const& x) override {
        std::array<std::array<double, 3>, 6> reference = {{{0.0, 0.0, 0.0},
                                                           {1.0, 0.0, 0.0},
                                                           {1.0, 1.0, 0.0},
                                                           {0.0, 1.0, 0.0},
                                                           {2.0, 0.0, 0.0},
                                                           {2.0, 1.0, 0.0}}};
        REQUIRE(id < 6);
        for (std::size_t i = 0; i < 3; ++i) {
            CHECK(x[i] == doctest::Approx(reference[id][i]));
        }
    }
    void setNumElements(std::size_t numElements) override { CHECK(numElements == 2); }
    void addElement(long type, int tag, long* node, std::size_t numNodes) override {
        REQUIRE(elNo < 2);
        if (elNo == 0) {
            CHECK(type == 3);
            CHECK(tag == 99);
            REQUIRE(numNodes == 4);
            CHECK(node[0] == 0);
            CHECK(node[1] == 1);
            CHECK(node[2] == 2);
            CHECK(node[3] == 3);
        } else {
            CHECK(type == 1);
            CHECK(tag == 97);
            REQUIRE(numNodes == 2);
            CHECK(node[0] == 5);
            CHECK(node[1] == 4);
        }
        ++elNo;
    }
};

TEST_CASE("GMSH") {
    char msh[] = R"MSH($MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
6
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 2.0 0.0 0.0
6 2.0 1.0 0.0
$EndNodes
$Elements
2
1 3 2 99 2 1 2 3 4
2 1 1 97 6 5
$EndElements
$NodeData
1
"A scalar view"
1
0.0
3
0
1
6
1 0.0
2 0.1
3 0.2
4 0.0
5 0.2
6 0.4
$EndNodeData
)MSH";

    MyTestBuilder builder;
    GMSHParser parser(&builder);
    bool ok = parser.parse(msh, sizeof(msh));
    if (!ok) {
        std::cout << parser.getErrorMessage();
    }
}
