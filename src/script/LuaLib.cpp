#include "LuaLib.h"
#include <iostream>

extern "C" {
#include <lualib.h>
#include <lauxlib.h>
}

namespace tndm {

LuaLib::LuaLib() {
    L = luaL_newstate();
    luaL_openlibs(L);
}

LuaLib::~LuaLib() {
    lua_close(L);
}

void LuaLib::load(std::string const& code) {
    int error = luaL_loadbuffer(L, code.c_str(), code.length(), "code") || lua_pcall(L, 0, 0, 0);
    if (error) {
        std::cerr << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1); 
    }
}

void LuaLib::loadFile(std::string const& fileName) {
    int error = luaL_loadfile(L, fileName.c_str()) || lua_pcall(L, 0, 0, 0);
    if (error) {
        std::cerr << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1); 
    }
}

} // namespace tndm
