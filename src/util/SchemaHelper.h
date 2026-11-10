#ifndef SCHEMAHELPER_20200930_H
#define SCHEMAHELPER_20200930_H

#include <algorithm>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <iostream>

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
        auto p = std::filesystem::path(path);
        if (!p.has_parent_path()) {
            return true;
        }
        return std::filesystem::exists(p.parent_path());
    }
};

/**
 * @brief Accepts a path whose parent directory exists or can be created.
 *
 * This holds if the nearest existing ancestor of the parent directory is a directory.
 * The directories are not created here (validators run on every MPI rank); see
 * create_output_directories in SEAS.cpp.
 */
class ParentPathExistsOrCreatable {
public:
    bool operator()(std::string const& path) {
        namespace fs = std::filesystem;
        auto p = fs::path(path);
        if (!p.has_parent_path()) {
            return true;
        }
        std::error_code ec;
        auto dir = p.parent_path();
        while (!dir.empty() && !fs::exists(dir, ec)) {
            auto parent = dir.parent_path();
            if (parent == dir) {
                break;
            }
            dir = std::move(parent);
        }
        // dir is empty (relative path, nothing above exists, i.e. the working directory)
        // or the nearest existing ancestor, which must be a directory.
        return dir.empty() || fs::is_directory(dir, ec);
    }
};

class MakePathRelativeToOtherPath {
public:
    template <typename Fun> MakePathRelativeToOtherPath(Fun otherPath) : otherPath_(otherPath) {}

    auto operator()(std::string_view path) {
        auto p = std::filesystem::path(path);
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
