#ifndef CSVWRITER_20211118_H
#define CSVWRITER_20211118_H

#include "TableWriter.h"

#include <fstream>
#include <iomanip>
#include <ios>
#include <string>
#include <string_view>

namespace tndm {

class CSVWriter : public TableWriter {
protected:
    template <typename T> void add(T const& t) { out_ << t << sep_; }

    std::ofstream out_;

private:
    char sep_ = ' ';

public:
    CSVWriter(char separator = ',', int precision = 15) : out_{}, sep_(separator) {
        out_ << std::scientific;
        set_precision(precision);
    }
    virtual ~CSVWriter() {}

    inline void set_separator(char separator) { sep_ = separator; }
    inline void set_precision(int precision) { out_ << std::setprecision(precision); }

    inline std::string_view default_extension() const override { return ".csv"; }
    inline void open(std::string const& file_name, bool append) override {
        auto openmode = std::ios::out;
        if (append) {
            /* We use ios::ate instead of ios::app, because with ios::app we cannot use
             * seekp. ios::in needs to be added for some reason because otherwise the
             * file is overwritten.
             */
            openmode |= std::ios::ate | std::ios::in;
        }
        out_.open(file_name, openmode);
    }
    inline void close() override {
        out_.close();
        out_.clear();
    }
    inline void add_title(std::string_view title) override { out_ << "# " << title << std::endl; }
    inline void endrow() override {
        out_.seekp(-1, std::ios::cur);
        out_ << std::endl;
    }
    inline void beginheader() override {}
    inline void endheader() override { endrow(); }
    inline void flush() override { out_.flush(); }

    inline TableWriter& operator<<(int value) override {
        add(value);
        return *this;
    }
    inline TableWriter& operator<<(unsigned int value) override {
        add(value);
        return *this;
    }
    inline TableWriter& operator<<(float value) override {
        add(value);
        return *this;
    }
    inline TableWriter& operator<<(double value) override {
        add(value);
        return *this;
    }
    inline TableWriter& operator<<(std::string_view value) override {
        out_ << "\"" << value << "\"" << sep_;
        return *this;
    }
};

} // namespace tndm

#endif // CSVWRITER_20211118_H
