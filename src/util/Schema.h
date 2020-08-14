#ifndef SCHEMA_20200812_H
#define SCHEMA_20200812_H

#include <stdexcept>
#include <toml.hpp>

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

namespace tndm {

template <typename Derived> struct SchemaTraits;
template <template <typename> typename Derived, typename T> struct SchemaTraits<Derived<T>> {
    using value_type = T;
};

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

    void print_help(std::ostream& out) const { out << help_; }

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
            throw std::runtime_error("Value type could not be converted to schema type");
        }
        if (!this->validator_(*result)) {
            throw std::runtime_error("Validator returned false");
        }
        return result.value();
    }
};

template <typename T> class ArraySchema;
template <typename T> class TableSchema;
template <typename T> class ArraySchemaCommon : public SchemaCommon<ArraySchemaCommon<T>> {
public:
    using value_type = typename T::value_type;

    template <template <typename> typename S> auto& of() {
        auto model = std::make_unique<Model<S>>();
        auto& schema = model->schema();
        of_ = std::move(model);
        return schema;
    }

    auto& of_values() { return of<ValueSchema>(); }
    auto& of_arrays() { return of<ArraySchema>(); }
    auto& of_tables() { return of<TableSchema>(); }

    virtual void print_help(std::ostream& out) const {
        out << "[";
        if (of_) {
            of_->print_help(out);
        }
        out << "]";
    }

protected:
    void commonChecks() const {
        if (!this->of_) {
            throw std::runtime_error("Missing value schema in array schema");
        }
    }

    auto get_array(toml::node_view<const toml::node> node) const {
        toml::array const* raw = node.as_array();
        if (!raw) {
            throw std::runtime_error("Expected array");
        }
        return raw;
    }

    class Concept {
    public:
        virtual ~Concept() {}
        virtual void translate(value_type& entry, toml::node_view<const toml::node> node) const = 0;
        virtual void print_help(std::ostream& out) const = 0;
    };

    template <template <typename> typename S> class Model : public Concept {
    public:
        void translate(value_type& entry, toml::node_view<const toml::node> node) const override {
            entry = schema_.translate(node);
        }
        void print_help(std::ostream& out) const override { schema_.print_help(out); }
        S<value_type>& schema() { return schema_; }

    private:
        S<value_type> schema_;
    };

    std::unique_ptr<Concept> of_;
};

template <typename T> class ArraySchema : public ArraySchemaCommon<T> {
public:
    ArraySchema<T>& min(std::size_t m) {
        min_ = m;
        return *this;
    }
    ArraySchema<T>& max(std::size_t m) {
        max_ = m;
        return *this;
    }

    auto translate(toml::node_view<const toml::node> node) const {
        this->commonChecks();
        auto raw = this->get_array(node);
        if (raw->size() < min_) {
            std::stringstream ss;
            ss << "Array too small (min size = " << min_ << ")";
            throw std::runtime_error(ss.str());
        }
        if (raw->size() > max_) {
            std::stringstream ss;
            ss << "Array too large (max size = " << max_ << ")";
            throw std::runtime_error(ss.str());
        }
        T vector(raw->size());
        for (std::size_t idx = 0; idx < raw->size(); ++idx) {
            this->of_->translate(vector[idx], node[idx]);
        }
        return vector;
    }

private:
    std::size_t min_ = 0, max_ = std::numeric_limits<std::size_t>::max();
};

template <typename T, std::size_t N>
class ArraySchema<std::array<T, N>> : public ArraySchemaCommon<std::array<T, N>> {
public:
    auto translate(toml::node_view<const toml::node> node) const {
        this->commonChecks();
        auto raw = this->get_array(node);
        if (raw->size() != N) {
            throw std::runtime_error("Array size mismatch");
        }
        std::array<T, N> array;
        for (std::size_t idx = 0; idx < raw->size(); ++idx) {
            this->of_->translate(array[idx], node[idx]);
        }
        return array;
    }
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
            } catch (std::runtime_error const& e) {
                throw std::runtime_error("Error while checking \"" + key + "\": " + e.what());
            }
        }
        return table;
    }

    virtual void print_help(std::ostream& out) const {
        if (!this->help_.empty()) {
            out << this->help_ << std::endl;
        }
        for (auto&& [key, schema] : entries_) {
            out << key;
            out << ": ";
            schema->print_help(out);
            out << std::endl;
        }
    }

private:
    class Concept {
    public:
        virtual ~Concept() {}
        virtual void translate(T& table, toml::node_view<const toml::node> node) const = 0;
        virtual void print_help(std::ostream& out) const = 0;
    };

    template <typename U, template <typename> typename S> class Model : public Concept {
    public:
        Model(U T::*member) : member_(member) {}
        void translate(T& table, toml::node_view<const toml::node> node) const override {
            if (!node) {
                if (!schema_.get_default_value()) {
                    throw std::runtime_error(
                        "Value missing although non-optional and no default is provided");
                }
                table.*member_ = *schema_.get_default_value();
                return;
            }
            table.*member_ = schema_.translate(node);
        }
        void print_help(std::ostream& out) const override {
            schema_.print_help(out);
            out << " (*)";
        }
        S<U>& schema() { return schema_; }

    private:
        U T::*member_;
        S<U> schema_;
    };

    template <typename U, template <typename> typename S>
    class Model<std::optional<U>, S> : public Concept {
    public:
        Model(std::optional<U> T::*member) : member_(member) {}
        void translate(T& table, toml::node_view<const toml::node> node) const override {
            if (node) {
                table.*member_ = std::make_optional(schema_.translate(node));
            }
        }
        void print_help(std::ostream& out) const override { schema_.print_help(out); }
        S<U>& schema() { return schema_; }

    private:
        std::optional<U> T::*member_;
        S<U> schema_;
    };

    std::vector<std::pair<std::string, std::unique_ptr<Concept>>> entries_;
};

} // namespace tndm

template <typename T> std::ostream& operator<<(std::ostream& lhs, tndm::TableSchema<T> const& rhs) {
    rhs.print_help(lhs);
    return lhs;
}

#endif // SCHEMA_20200812_H
