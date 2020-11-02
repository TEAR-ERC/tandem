#ifndef LUA_20200811_H
#define LUA_20200811_H

#include <stdexcept>
extern "C" {
#include <lua.h>
}

#include <array>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tndm {

class LuaLib {
public:
    template <std::size_t Din, std::size_t Dout>
    using functional_t = std::function<std::array<double, Dout>(std::array<double, Din> const& x)>;

    LuaLib();
    ~LuaLib();

    void load(std::string const& code);
    void loadFile(std::string const& fileName);

    template <int Din, int Dout> auto getFunction(std::string const& name) {
        lua_State* myL = L;
        return [myL, name](std::array<double, Din> const& x) -> std::array<double, Dout> {
            std::array<double, Dout> result;
            result.fill(std::numeric_limits<double>::signaling_NaN());

            lua_getglobal(myL, name.c_str());
            for (int d = 0; d < Din; ++d) {
                lua_pushnumber(myL, x[d]);
            }
            int error = lua_pcall(myL, Din, Dout, 0);
            if (error) {
                std::stringstream ss;
                ss << "error running '" << name << "': " << lua_tostring(myL, -1);
                throw std::runtime_error(ss.str());
            }

            for (int d = 0; d < Dout; ++d) {
                if (!lua_isnumber(myL, d - Dout)) {
                    std::stringstream ss;
                    ss << "'" << name << "' returned not a number (" << d << ").";
                    throw std::runtime_error(ss.str());
                }
                result[d] = lua_tonumber(myL, d - Dout);
            }
            lua_pop(myL, Dout);
            return result;
        };
    }

private:
    lua_State* L;
};

} // namespace tndm

#endif // LUA_20200811_H
