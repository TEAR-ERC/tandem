#include "doctest.h"
#include "mesh/Simplex.h"

using tndm::Simplex;

TEST_CASE("Simplex") {
    Simplex<3> plex({9,6,4,5});

    SUBCASE("access") {
        CHECK(plex[0] == 4);
        CHECK(plex[1] == 5);
        CHECK(plex[2] == 6);
        CHECK(plex[3] == 9);
    }

    SUBCASE("downward 0") {
        auto dws = plex.downward<0>();
        CHECK(dws[0][0] == 4);
        CHECK(dws[1][0] == 5);
        CHECK(dws[2][0] == 6);
        CHECK(dws[3][0] == 9);
    }

    SUBCASE("downward 1") {
        auto dws = plex.downward<1>();
        int edges[][2] = {{4,5}, {4,6}, {4,9}, {5,6}, {5,9}, {6,9}};
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 2; ++j) {
                CHECK(dws[i][j] == edges[i][j]);
            }
        }
    }

    SUBCASE("downward 2") {
        auto dws = plex.downward();
        int facets[][3] = {{4,5,6}, {4,5,9}, {4,6,9}, {5,6,9}};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                CHECK(dws[i][j] == facets[i][j]);
            }
        }
    }
}
