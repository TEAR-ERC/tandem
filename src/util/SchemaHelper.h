#ifndef SCHEMAHELPER_20200930_H
#define SCHEMAHELPER_20200930_H

#include <algorithm>
#ifdef EXPERIMENTAL_FS
#include <experimental/filesystem>
#else
#include <filesystem>
#endif
#include <functional>
#include <string>
#include <string_view>

#include <iostream>

#ifdef EXPERIMENTAL_FS
namespace fs = std::experimental::filesystem;
#else
namespace fs = std::filesystem;
#endif

namespace tndm {

class PathExists {
public:
    bool operator()(std::string const& path) {
        return std::filesystem::exists(std::filesystem::path(path));
    }
};

class ParentPathExists {
public:
    bool operator()(std::string const& path) {
        auto p = fs::path(path);
        if (!p.has_parent_path()) {
            return true;
        }
        return std::filesystem::exists(p.parent_path());
    }
};

class MakePathRelativeToOtherPath {
public:
    template <typename Fun> MakePathRelativeToOtherPath(Fun otherPath) : otherPath_(otherPath) {}

    auto operator()(std::string_view path) {
        auto p = fs::path(path);
        if (p.is_relative()) {
            auto newPath = std::filesystem::path(otherPath_()).parent_path();
            newPath /= p;
            return newPath;
        }
        return p;
    }

private:
    std::function<std::string(void)> otherPath_;
};

inline bool iEquals(std::string_view a, std::string_view b) {
    return std::equal(a.begin(), a.end(), b.begin(),
                      [](char a, char b) { return tolower(a) == tolower(b); });
}

} // namespace tndm

#endif // SCHEMAHELPER_20200930_H
