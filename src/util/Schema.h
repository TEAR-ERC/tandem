#ifndef SCHEMA_20200812_H
#define SCHEMA_20200812_H

#include "Traits.h"

#include <stdexcept>
#include <toml.hpp>

#include <charconv>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace tndm {

template <typename Derived> struct SchemaTraits;
template <template <typename> typename Derived, typename T> struct SchemaTraits<Derived<T>> {
    using value_type = T;
};

template <typename T> class ValueSchema;
template <typename T> class ArraySchema;
template <typename T> class TableSchema;

template <typename Derived> class SchemaCommon {
public:
    using value_type = typename SchemaTraits<Derived>::value_type;
    using validator_fun_t = std::function<bool(value_type const& value)>;

    Derived& help(std::string&& help) {
        help_ = std::move(help);
        return static_cast<Derived&>(*this);
    }

    Derived& validator(validator_fun_t&& validator) {
        validator_ = std::move(validator);
        return static_cast<Derived&>(*this);
    }

    Derived& default_value(value_type&& val) {
        default_ = std::optional<value_type>(std::move(val));
        return static_cast<Derived&>(*this);
    }

    auto const& get_default_value() const { return default_; }
    std::string_view get_help() const { return help_; }

protected:
    std::string help_;
    validator_fun_t validator_ = [](value_type const& value) { return true; };
    std::optional<value_type> default_ = std::nullopt;
};

template <typename T> class ValueSchema : public SchemaCommon<ValueSchema<T>> {
public:
    T translate(toml::node_view<const toml::node> node) const {
        auto result = node.value<T>();
        if (!result) {
            std::stringstream ss;
            ss << "Value " << node << " could not be converted to expected type " << type_name();
            throw std::runtime_error(ss.str());
        }
        if (!this->validator_(*result)) {
            std::stringstream ss;
            throw std::runtime_error("Validator returned false");
        }
        return result.value();
    }

    T translate(std::string_view value) const {
        T result;
        if constexpr (std::is_same_v<bool, T>) {
            std::string data;
            std::transform(value.begin(), value.end(), data.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            result = data == "yes" || data == "true";
            if (!result) {
                if (data != "no" && data != "false") {
                    std::stringstream ss;
                    ss << "Could not convert " << value << " to boolean";
                    throw std::runtime_error(ss.str());
                }
            }
        } else if constexpr (std::is_integral<T>() || std::is_floating_point<T>()) {
            std::from_chars_result res =
                std::from_chars(value.begin(), value.end() + value.size(), result);
            if (res.ec == std::errc::invalid_argument) {
                throw std::invalid_argument{"Invalid argument"};
            } else if (res.ec == std::errc::result_out_of_range) {
                throw std::out_of_range{"Out of range"};
            }
        } else if constexpr (is_string<T>()) {
            result = value;
        }
        if (!this->validator_(result)) {
            std::stringstream ss;
            throw std::runtime_error("Validator returned false");
        }
        return result;
    }

    void print_schema(std::ostream& out, bool) const {
        if (this->default_) {
            if constexpr (is_string<T>()) {
                out << "\"" << *this->default_ << "\"";
            } else {
                out << *this->default_;
            }
        } else {
            out << type_name();
        }
    }

private:
    std::string type_name() const {
        if constexpr (std::is_same_v<bool, T>) {
            return "<boolean>";
        } else if constexpr (std::is_integral<T>()) {
            return "<integer>";
        } else if constexpr (std::is_floating_point<T>()) {
            return "<float>";
        } else if constexpr (is_string<T>()) {
            return "<string>";
        }
    }
};

template <typename Derived> class ArraySchemaCommon : public SchemaCommon<Derived> {
public:
    ArraySchemaCommon() : min_(0), max_(std::numeric_limits<std::size_t>::max()) {}
    ArraySchemaCommon(std::size_t min, std::size_t max) : min_(min), max_(max) {}

    using array_type = typename SchemaTraits<Derived>::value_type;
    using value_type = typename array_type::value_type;

    template <template <typename> typename S> auto& of() {
        auto model = std::make_unique<Model<S>>();
        auto& schema = model->schema();
        of_ = std::move(model);
        return schema;
    }

    auto& of_values() { return of<ValueSchema>(); }
    auto& of_arrays() { return of<ArraySchema>(); }
    auto& of_tables() { return of<TableSchema>(); }

    auto translate(toml::node_view<const toml::node> node) const {
        if (!this->of_) {
            throw std::runtime_error("Missing value schema in array schema");
        }
        toml::array const* raw = node.as_array();
        if (!raw) {
            throw std::runtime_error("Expected array");
        }
        if (raw->size() < this->min_ || raw->size() > this->max_) {
            std::stringstream ss;
            ss << "Given array size is n = " << raw->size() << "; should be " << this->min_
               << " <= n <= " << this->max_;
            throw std::runtime_error(ss.str());
        }
        if (raw->size() > this->max_) {
            std::stringstream ss;
            ss << "Array too large (max size = " << this->max_ << ")";
            throw std::runtime_error(ss.str());
        }
        auto array = static_cast<Derived const*>(this)->make(raw->size());
        for (std::size_t idx = 0; idx < raw->size(); ++idx) {
            try {
                this->of_->translate(array[idx], node[idx]);
            } catch (std::runtime_error const& e) {
                std::stringstream ss;
                ss << e.what() << std::endl;
                ss << "  --> entry " << idx;
                auto help = of_->get_help();
                if (!help.empty()) {
                    ss << " (\"" << help << "\")";
                }
                throw std::runtime_error(ss.str());
            }
        }
        return array;
    }

    array_type translate(std::string_view value) const {
        throw std::logic_error{"Not implemented"};
    }

    void print_schema(std::ostream& out, bool) const {
        out << "[";
        if (min_ == max_) {
            out << min_;
        } else if (max_ == std::numeric_limits<std::size_t>::max()) {
            out << min_ << "--" << max_;
        }
        out << " x ";
        if (of_) {
            of_->print_schema(out, true);
        }
        out << "]";
    }

protected:
    class Concept {
    public:
        virtual ~Concept() {}
        virtual void translate(value_type& entry, toml::node_view<const toml::node> node) const = 0;
        virtual void print_schema(std::ostream& out, bool inLine) const = 0;
        virtual std::string_view get_help() const = 0;
    };

    template <template <typename> typename S> class Model : public Concept {
    public:
        void translate(value_type& entry, toml::node_view<const toml::node> node) const override {
            entry = schema_.translate(node);
        }
        void print_schema(std::ostream& out, bool inLine) const override {
            schema_.print_schema(out, inLine);
        }
        std::string_view get_help() const override { return schema_.get_help(); }
        S<value_type>& schema() { return schema_; }

    private:
        S<value_type> schema_;
    };

    std::unique_ptr<Concept> of_;
    std::size_t min_, max_;
};

template <typename T> class ArraySchema : public ArraySchemaCommon<ArraySchema<T>> {
public:
    ArraySchema<T>& min(std::size_t m) {
        this->min_ = m;
        return *this;
    }
    ArraySchema<T>& max(std::size_t m) {
        this->max_ = m;
        return *this;
    }
    auto make(std::size_t size) const { return T(size); }
};

template <typename T, std::size_t N>
class ArraySchema<std::array<T, N>> : public ArraySchemaCommon<ArraySchema<std::array<T, N>>> {
public:
    ArraySchema() : ArraySchemaCommon<ArraySchema<std::array<T, N>>>(N, N) {}
    auto make(std::size_t) const { return std::array<T, N>{}; }
};

template <typename T> class TableSchema : public SchemaCommon<TableSchema<T>> {
public:
    template <typename U, template <typename> typename S>
    auto& add(std::string&& name, U T::*member) {
        auto entry = std::make_unique<Model<U, S>>(member);
        auto& schema = entry->schema();
        entries_.emplace_back(std::make_pair(std::move(name), std::move(entry)));
        return schema;
    }

    template <typename U> auto& add_value(std::string&& name, U T::*member) {
        return add<U, ValueSchema>(std::move(name), member);
    }

    template <typename U> auto& add_array(std::string&& name, U T::*member) {
        return add<U, ArraySchema>(std::move(name), member);
    }

    template <typename U> auto& add_table(std::string&& name, U T::*member) {
        return add<U, TableSchema>(std::move(name), member);
    }

    T translate(toml::node_view<const toml::node> node) const {
        T table;
        for (auto&& [key, model] : entries_) {
            try {
                model->translate(table, node[key]);
            } catch (std::exception const& e) {
                throw std::runtime_error(format_error(e, key, *model));
            }
        }
        return table;
    }

    T translate(std::string_view value) const { throw std::logic_error{"Not implemented"}; }

    void set(T& table, std::string_view key, std::string_view value) {
        for (auto&& [k, model] : entries_) {
            if (key == k) {
                try {
                    model->translate(table, value);
                } catch (std::exception const& e) {
                    throw std::runtime_error(format_error(e, key, *model));
                }
                return;
            }
        }
    }

    void print_schema(std::ostream& out, bool inLine) const {
        if (inLine) {
            out << "{";
            std::size_t num = 0;
            for (auto&& [key, schema] : entries_) {
                out << key;
                out << " = ";
                schema->print_schema(out, true);
                if (++num < entries_.size()) {
                    out << ", ";
                }
            }
            out << "}";
        } else {
            for (auto&& [key, schema] : entries_) {
                out << key;
                out << " = ";
                schema->print_schema(out, true);
                out << std::endl;
            }
        }
    }

    void cmd_line_args(std::function<void(std::string_view, std::string_view)> callback) const {
        for (auto&& [key, schema] : entries_) {
            schema->cmd_line_args(key, callback);
        }
    }

private:
    class Concept {
    public:
        virtual ~Concept() {}
        virtual void translate(T& table, toml::node_view<const toml::node> node) const = 0;
        virtual void translate(T& table, std::string_view value) const = 0;
        virtual void print_schema(std::ostream& out, bool inLine) const = 0;
        virtual std::string_view get_help() const = 0;
        virtual void
        cmd_line_args(std::string_view key,
                      std::function<void(std::string_view, std::string_view)> callback) const = 0;
    };

    template <typename U, template <typename> typename S> class ModelCommon : public Concept {
    public:
        void print_schema(std::ostream& out, bool inLine) const override {
            schema_.print_schema(out, inLine);
        }
        std::string_view get_help() const override { return schema_.get_help(); }
        S<U>& schema() { return schema_; }
        void cmd_line_args(
            std::string_view key,
            std::function<void(std::string_view, std::string_view)> callback) const override {
            if constexpr (std::is_same_v<S<U>, ValueSchema<U>>) {
                callback(key, schema_.get_help());
            }
        }

    protected:
        S<U> schema_;
    };

    template <typename U, template <typename> typename S> class Model : public ModelCommon<U, S> {
    public:
        Model(U T::*member) : member_(member) {}
        void translate(T& table, toml::node_view<const toml::node> node) const override {
            if (!node) {
                if (!this->schema_.get_default_value()) {
                    throw std::runtime_error(
                        "Value missing although non-optional and no default is provided");
                }
                table.*member_ = *this->schema_.get_default_value();
                return;
            }
            table.*member_ = this->schema_.translate(node);
        }
        void translate(T& table, std::string_view value) const override {
            table.*member_ = this->schema_.translate(value);
        }

    private:
        U T::*member_;
    };

    template <typename U, template <typename> typename S>
    class Model<std::optional<U>, S> : public ModelCommon<U, S> {
    public:
        Model(std::optional<U> T::*member) : member_(member) {}
        void translate(T& table, toml::node_view<const toml::node> node) const override {
            if (node) {
                table.*member_ = std::make_optional(this->schema_.translate(node));
            }
        }
        void translate(T& table, std::string_view value) const override {
            table.*member_ = std::make_optional(this->schema_.translate(value));
        }

    private:
        std::optional<U> T::*member_;
    };

    std::string format_error(std::exception const& e, std::string_view key,
                             Concept const& model) const {
        std::stringstream ss;
        ss << e.what() << std::endl;
        ss << "  --> " << key;
        auto help = model.get_help();
        if (!help.empty()) {
            ss << " (\"" << help << "\")";
        }
        return ss.str();
    }

    std::vector<std::pair<std::string, std::unique_ptr<Concept>>> entries_;
};

} // namespace tndm

template <typename T> std::ostream& operator<<(std::ostream& lhs, tndm::TableSchema<T> const& rhs) {
    rhs.print_schema(lhs, false);
    return lhs;
}

#endif // SCHEMA_20200812_H
