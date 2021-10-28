#include <iosfwd>

namespace tndm {

template <class CharT, class Traits = std::char_traits<CharT>>
class basic_nullbuf : public std::basic_streambuf<CharT, Traits> {
    auto overflow(typename Traits::int_type c) -> typename Traits::int_type {
        return Traits::not_eof(c);
    }
};

template <class CharT, class Traits = std::char_traits<CharT>>
class basic_nullostream : public std::basic_ostream<CharT, Traits> {
public:
    basic_nullostream() : std::basic_ostream<CharT, Traits>(&buf_) {}

private:
    basic_nullbuf<CharT, Traits> buf_;
};

using nullostream = basic_nullostream<char>;

} // namespace tndm
