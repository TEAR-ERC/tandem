#include "doctest.h"
#include "script/LuaLib.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <memory>

using tndm::LuaLib;

TEST_CASE("Lua functions") {
    LuaLib lua;
    lua.load(R"LUA(
function f(x,y)
    return x * math.cos(y), x * math.sin(y)
end
)LUA");

    auto f = lua.getFunction<2, 2>("f");
    auto [fx, fy] = f({2.0, M_PI / 4.0});
    CHECK(fx == doctest::Approx(sqrt(2.0)));
    CHECK(fy == doctest::Approx(sqrt(2.0)));
}
