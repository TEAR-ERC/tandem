#ifndef SCHEMAHELPER_20200930_H
#define SCHEMAHELPER_20200930_H

#include <algorithm>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

namespace tndm {

class PathExists {
public:
    bool operator()(std::string const& path) {
        return std::filesystem::exists(std::filesystem::path(path));
    }
};

class MakePathRelativeToOtherPath {
public:
    template <typename Fun> MakePathRelativeToOtherPath(Fun otherPath) : otherPath_(otherPath) {}

    auto operator()(std::string_view path) {
        auto newPath = std::filesystem::path(otherPath_()).parent_path();
        newPath /= std::filesystem::path(path);
        return newPath;
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
