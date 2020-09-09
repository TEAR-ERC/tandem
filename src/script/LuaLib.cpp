#include "LuaLib.h"
#include <filesystem>
#include <iostream>

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}

namespace fs = std::filesystem;

namespace tndm {

LuaLib::LuaLib() {
    L = luaL_newstate();
    luaL_openlibs(L);
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
    auto fileNamePath = fs::canonical(fs::absolute(fs::path(fileName).parent_path())).string();
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
