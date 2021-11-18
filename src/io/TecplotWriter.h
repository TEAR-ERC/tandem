#ifndef TECPLOTWRITER_20211118_H
#define TECPLOTWRITER_20211118_H

#include "CSVWriter.h"

#include <string_view>

namespace tndm {

class TecplotWriter : public CSVWriter {
public:
    TecplotWriter(int precision = 15) : CSVWriter(' ', precision) {}
    virtual ~TecplotWriter() {}

    inline std::string_view default_extension() const override { return ".dat"; }
    inline void add_title(std::string_view title) override {
        out_ << "TITLE = \"" << title << "\"" << std::endl;
    }
    inline void beginheader() override {
        out_ << "VARIABLES = ";
        set_separator(',');
    }
    inline void endheader() override {
        endrow();
        set_separator(' ');
    }
};

} // namespace tndm

#endif // TECPLOTWRITER_20211118_H
