#ifndef SCHEMA_20200812_H
#define SCHEMA_20200812_H

#include <stdexcept>
#include <toml.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

namespace tndm {

class SchemaBase {
public:
    virtual ~SchemaBase() {}
    virtual void translate(toml::node_view<const toml::node> node) = 0;
    virtual void print_help(std::ostream& out) const = 0;
    virtual bool is_value() const { return false; }

    bool is_required() const { return required_; }

protected:
    bool required_ = false;
};

std::ostream& operator<<(std::ostream& lhs, SchemaBase const& rhs) {
    rhs.print_help(lhs);
    return lhs;
}

template <typename T> class ValueSchema : public SchemaBase {
public:
    using validator_fun_t = std::function<bool(T const& value)>;

    ValueSchema& required() {
        required_ = true;
        return *this;
    }

    ValueSchema& help(std::string&& help) {
        help_ = std::move(help);
        return *this;
    }

    ValueSchema& validator(validator_fun_t&& validator) {
        validator_ = std::move(validator);
        return *this;
    }

    void translate(toml::node_view<const toml::node> node) {
        if (!node) {
            if (required_) {
                throw std::runtime_error("Value missing although required");
            }
            return;
        }
        if (!node.is_value() || !node.is<T>()) {
            throw std::runtime_error("Value type does not match schema");
        }
        if (!validator_(node.as<T>()->get())) {
            throw std::runtime_error("Validator returned false");
        }
    }

    virtual void print_help(std::ostream& out) const {
        out << help_;
    }

    virtual bool is_value() const { return true; }

private:
    std::string help_;
    validator_fun_t validator_ = [](T const& value) { return true; };
};

class TableSchema : public SchemaBase {
public:
    TableSchema() {}

    template <typename T> ValueSchema<T>& add_value(std::string&& name) {
        entries_.emplace_back(std::make_pair(std::move(name), std::make_unique<ValueSchema<T>>()));
        return static_cast<ValueSchema<T>&>(*entries_.back().second);
    }

    void translate(toml::node_view<const toml::node> node) {
        for (auto&& [key, schema] : entries_) {
            try {
                schema->translate(node[key]);
            } catch (std::runtime_error const& e) {
                throw std::runtime_error("Error while checking \"" + key + "\": " + e.what());
            }
        }
    }

    virtual void print_help(std::ostream& out) const {
        for (auto&& [key, schema] : entries_) {
            out << key;
            if (schema->is_required()) {
                out << " (*)";
            }
            out << ": " << *schema << std::endl;
        }
    }

private:
    std::vector<std::pair<std::string, std::unique_ptr<SchemaBase>>> entries_;
};

} // namespace tndm

#endif // SCHEMA_20200812_H
