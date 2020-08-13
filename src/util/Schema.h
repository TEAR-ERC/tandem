#ifndef SCHEMA_20200812_H
#define SCHEMA_20200812_H

#include <stdexcept>
#include <toml.hpp>

#include <functional>
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

template <typename T> class TableSchema : public SchemaCommon<TableSchema<T>> {
public:
    TableSchema() {}

    template <typename U> auto& add_value(std::string&& name, U T::*member) {
        auto entry = std::make_unique<EntryModel<U, ValueSchema>>(member);
        auto& schema = entry->schema();
        entries_.emplace_back(std::make_pair(std::move(name), std::move(entry)));
        return schema;
    }

    template <typename U> auto& add_table(std::string&& name, U T::*member) {
        auto entry = std::make_unique<EntryModel<U, TableSchema>>(member);
        auto& schema = entry->schema();
        entries_.emplace_back(std::make_pair(std::move(name), std::move(entry)));
        return schema;
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
    class EntryConcept {
    public:
        virtual ~EntryConcept() {}
        virtual void translate(T& table, toml::node_view<const toml::node> node) const = 0;
        virtual void print_help(std::ostream& out) const = 0;
    };

    template <typename U, template <typename> typename S> class EntryModel : public EntryConcept {
    public:
        EntryModel(U T::*member) : member_(member) {}
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
    class EntryModel<std::optional<U>, S> : public EntryConcept {
    public:
        EntryModel(std::optional<U> T::*member) : member_(member) {}
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

    std::vector<std::pair<std::string, std::unique_ptr<EntryConcept>>> entries_;
};

} // namespace tndm

template <typename T> std::ostream& operator<<(std::ostream& lhs, tndm::TableSchema<T> const& rhs) {
    rhs.print_help(lhs);
    return lhs;
}

#endif // SCHEMA_20200812_H
