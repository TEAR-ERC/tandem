#include "LuaLib.h"
#include <cmath>
#ifdef EXPERIMENTAL_FS
#include <experimental/filesystem>
#else
#include <filesystem>
#endif
#include <iostream>

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>

static int math_acosh(lua_State* L) {
    lua_pushnumber(L, acosh(luaL_checknumber(L, 1)));
    return 1;
}

static int math_asinh(lua_State* L) {
    lua_pushnumber(L, asinh(luaL_checknumber(L, 1)));
    return 1;
}

static int math_atanh(lua_State* L) {
    lua_pushnumber(L, atanh(luaL_checknumber(L, 1)));
    return 1;
}
}

#ifdef EXPERIMENTAL_FS
namespace fs = std::experimental::filesystem;
#else
namespace fs = std::filesystem;
#endif

namespace tndm {

LuaLib::LuaLib() {
    L = luaL_newstate();
    luaL_openlibs(L);

    auto add_cfun = [this](char const* name, lua_CFunction fun) {
        lua_pushstring(this->L, name);
        lua_pushcfunction(this->L, fun);
        lua_settable(this->L, -3);
    };
    lua_getglobal(L, "math");
    add_cfun("acosh", math_acosh);
    add_cfun("asinh", math_asinh);
    add_cfun("atanh", math_atanh);
    lua_pop(L, 1);
}

LuaLib::~LuaLib() { lua_close(L); }

void LuaLib::load(std::string const& code) {
    int error = luaL_loadbuffer(L, code.c_str(), code.length(), "code") || lua_pcall(L, 0, 0, 0);
    if (error) {
        std::cerr << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
    }
}

void LuaLib::loadFile(std::string const& fileName) {
    lua_getglobal(L, "package");
    lua_getfield(L, -1, "path");
    std::string curPath = lua_tostring(L, -1);
    auto parent = fs::path(fileName).parent_path();
    if (parent.empty()) {
        parent = fs::current_path();
    }
    auto fileNamePath = fs::canonical(fs::absolute(parent)).string();
    curPath.append(";");
    curPath.append(fileNamePath);
    curPath.append("/?.lua");
    curPath.append(";");
    curPath.append(fileNamePath);
    curPath.append("/?/init.lua");
    lua_pop(L, 1);
    lua_pushstring(L, curPath.c_str());
    lua_setfield(L, -2, "path");
    lua_pop(L, 1);
    int error = luaL_loadfile(L, fileName.c_str()) || lua_pcall(L, 0, 0, 0);
    if (error) {
        std::cerr << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
    }
}

} // namespace tndm
