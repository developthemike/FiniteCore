#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "expr.h"

namespace fc {

// INI-style configuration:
//
//   # comment
//   title = Free text            (keys before the first section are global)
//   [section]
//   key = value                  (value may be any expression, see expr.h)
//   [boundary.hot]               (dotted section names group related sections)
//
// Keys that are never read are reported as warnings, which catches typos.
struct ConfigEntry {
    std::string key;
    std::string value;
    int line = 0;
    mutable bool used = false;
};

struct ConfigSection {
    std::string name;  // "" for the global section
    int line = 0;
    std::vector<ConfigEntry> entries;
    const ConfigEntry* find(const std::string& key) const;
};

using Constants = std::map<std::string, double>;

class Config {
public:
    static Config load(const std::string& path);

    // "section.key=value" (the key is the part after the last dot).
    void applyOverride(const std::string& spec);

    const ConfigSection* section(const std::string& name) const;
    // Sections named "<prefix>.<something>" in file order.
    std::vector<const ConfigSection*> sectionsWithPrefix(const std::string& prefix) const;
    // Prints warnings for keys that were never read. Returns the count.
    int warnUnused() const;

    std::string path;
    std::vector<ConfigSection> sections;
};

// Typed, error-reporting access to one section.
class SectionReader {
public:
    SectionReader(const Config& cfg, const ConfigSection* sec, const Constants& constants);

    bool exists() const { return sec_ != nullptr; }
    bool has(const std::string& key) const;
    const std::string& name() const;

    std::string str(const std::string& key, const std::string& def) const;
    std::string strRequired(const std::string& key) const;
    double number(const std::string& key, double def) const;  // constant expression
    double numberRequired(const std::string& key) const;
    int integer(const std::string& key, int def) const;
    bool boolean(const std::string& key, bool def) const;
    // Expression in x, y, t. `allowed` lists the permitted variables, e.g. "xy" or "xyt".
    Expr expr(const std::string& key, const std::string& def, const char* allowed) const;
    Expr exprRequired(const std::string& key, const char* allowed) const;
    // Comma-separated list of constant expressions.
    std::vector<double> numbers(const std::string& key) const;

    [[noreturn]] void fail(const std::string& key, const std::string& msg) const;
    std::string where(const std::string& key) const;

private:
    const ConfigEntry* entry(const std::string& key) const;
    const Config& cfg_;
    const ConfigSection* sec_;
    const Constants& constants_;
};

// Splits at commas that are not inside parentheses.
std::vector<std::string> splitTopLevel(const std::string& s, char sep = ',');
std::string trim(const std::string& s);
std::string toLower(std::string s);

}  // namespace fc
