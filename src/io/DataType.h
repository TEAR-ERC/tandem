#ifndef DATATYPE_20200630_H
#define DATATYPE_20200630_H

#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>

namespace tndm {

enum class BaseType { Float, Int, UInt, None };

class DataType {
public:
    DataType() : type_(BaseType::None), bytes_(0) {}
    DataType(BaseType type, unsigned bytes) : type_(type), bytes_(bytes) {}

    template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
    constexpr DataType(T) : type_(BaseType::Float), bytes_(sizeof(T)) {}

    template <typename T, std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>, int> = 0>
    constexpr DataType(T) : type_(BaseType::Int), bytes_(sizeof(T)) {}

    template <typename T, std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T>, int> = 0>
    constexpr DataType(T) : type_(BaseType::UInt), bytes_(sizeof(T)) {}

    BaseType type() const { return type_; }
    unsigned bytes() const { return bytes_; }

    std::string vtkIdentifier() const {
        std::stringstream s;
        switch (type_) {
        case BaseType::Float:
            s << "Float";
            break;
        case BaseType::Int:
            s << "Int";
            break;
        case BaseType::UInt:
            s << "UInt";
            break;
        case BaseType::None:
            s << "None";
            break;
        default:
            break;
        }
        s << bytes_ * 8;
        return s.str();
    }

private:
    BaseType type_;
    unsigned bytes_;
};

} // namespace tndm

#endif // DATATYPE_20200630_H
