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
    template <std::size_t Din, std::size_t Dout>
    using functional_t_region =
        std::function<std::array<double, Dout>(std::array<double, Din> const& x, long int& tag)>;

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

    bool hasMember(std::string const& table_name, char const* member_name) {
        lua_getglobal(L, table_name.c_str());
        if (lua_istable(L, -1) == 0) {
            throw std::runtime_error(table_name + " is not a table.");
        }
        lua_getfield(L, -1, member_name);
        bool ok = lua_isnil(L, -1) == 0;
        lua_pop(L, 2);
        return ok;
    }

    template <int Din, int Dout>
    auto getMemberFunction(std::string const& table_name, char const* method_name) {
        lua_State* myL = L;
        return [myL, table_name,
                method_name](std::array<double, Din> const& x) -> std::array<double, Dout> {
            std::array<double, Dout> result;
            result.fill(std::numeric_limits<double>::signaling_NaN());

            lua_getglobal(myL, table_name.c_str());
            if (lua_istable(myL, -1) == 0) {
                throw std::runtime_error(table_name + " is not a table.");
            }
            lua_getfield(myL, -1, method_name);
            lua_insert(myL, -2); // swap object and method as first argument is "self"
            for (int d = 0; d < Din; ++d) {
                lua_pushnumber(myL, x[d]);
            }
            int error = lua_pcall(myL, 1 + Din, Dout, 0);
            if (error) {
                std::stringstream ss;
                ss << "error running '" << table_name << "." << method_name
                   << "': " << lua_tostring(myL, -1);
                throw std::runtime_error(ss.str());
            }

            for (int d = 0; d < Dout; ++d) {
                if (!lua_isnumber(myL, d - Dout)) {
                    std::stringstream ss;
                    ss << "'" << table_name << "." << method_name << "' returned not a number ("
                       << d << ").";
                    throw std::runtime_error(ss.str());
                }
                result[d] = lua_tonumber(myL, d - Dout);
            }
            lua_pop(myL, Dout);
            return result;
        };
    }

    template <int Din, int Dout>
    auto getMemberFunctionTagged(std::string const& table_name, char const* method_name) {

        lua_State* myL = L;

        return [myL, table_name, method_name](std::array<double, Din> const& x,
                                              long int& tag) -> std::array<double, Dout> {
            std::array<double, Dout> result;
            result.fill(std::numeric_limits<double>::signaling_NaN());

            // Get table from Lua
            lua_getglobal(myL, table_name.c_str());
            if (!lua_istable(myL, -1)) {
                throw std::runtime_error(table_name + " " + method_name + " is not a table.");
            }

            // Get method from table
            lua_getfield(myL, -1, method_name); // push method
            lua_insert(myL, -2);

            // Push spatial coordinates
            for (int d = 0; d < Din; ++d) {
                lua_pushnumber(myL, x[d]);
            }

            // Push physicalTag as integer
            lua_pushinteger(myL, tag);

            // Call method: 1 (self) + Din + 1 (tag) = 1+Din+1
            int num_inputs = 1 + Din + 1;
            int error = lua_pcall(myL, num_inputs, Dout, 0);
            if (error) {
                std::stringstream ss;
                ss << "Error running '" << table_name << "." << method_name
                   << "': " << lua_tostring(myL, -1);
                throw std::runtime_error(ss.str());
            }

            // Retrieve outputs
            for (int d = 0; d < Dout; ++d) {
                if (!lua_isnumber(myL, d - Dout)) {
                    std::stringstream ss;
                    ss << "'" << table_name << "." << method_name
                       << "' returned non-number at index " << d;
                    throw std::runtime_error(ss.str());
                }
                result[d] = lua_tonumber(myL, d - Dout);
            }

            // Clean up stack
            lua_pop(myL, Dout);
            return result;
        };
    }

    double getMemberConstant(std::string const& table_name, std::string const& constant_name) {
        lua_getglobal(L, table_name.c_str());
        if (lua_istable(L, -1) == 0) {
            throw std::runtime_error(table_name + " is not a table.");
        }
        lua_getfield(L, -1, constant_name.c_str());
        if (!lua_isnumber(L, -1)) {
            std::stringstream ss;
            ss << "'" << table_name << "." << constant_name << "' returned not a number.";
            throw std::runtime_error(ss.str());
        }
        double result = lua_tonumber(L, -1);
        lua_pop(L, 2); // Remove table and field
        return result;
    }

private:
    lua_State* L;
};

} // namespace tndm

#endif // LUA_20200811_H
