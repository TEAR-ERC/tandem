#ifndef TABLEWRITER_20211118_H
#define TABLEWRITER_20211118_H

#include <cstddef>
#include <string_view>

namespace tndm {

class TableWriter {
public:
    virtual ~TableWriter() {}

    virtual std::string_view default_extension() const = 0;
    virtual void open(std::string const& file_name, bool append) = 0;
    virtual void close() = 0;
    virtual void add_title(std::string_view title) = 0;
    virtual void endrow() = 0;
    virtual void beginheader() = 0;
    virtual void endheader() = 0;
    virtual void flush() = 0;

    virtual TableWriter& operator<<(int value) = 0;
    virtual TableWriter& operator<<(unsigned int value) = 0;
    virtual TableWriter& operator<<(float value) = 0;
    virtual TableWriter& operator<<(double value) = 0;
    virtual TableWriter& operator<<(std::string_view value) = 0;
    virtual TableWriter& operator<<(TableWriter& (*func)(TableWriter&)) { return func(*this); }
};

inline TableWriter& endrow(TableWriter& writer) {
    writer.endrow();
    return writer;
}

inline TableWriter& beginheader(TableWriter& writer) {
    writer.beginheader();
    return writer;
}

inline TableWriter& endheader(TableWriter& writer) {
    writer.endheader();
    return writer;
}

} // namespace tndm

#endif // TABLEWRITER_20211118_H
