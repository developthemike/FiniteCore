#include "config.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fc {

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string toLower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::vector<std::string> splitTopLevel(const std::string& s, char sep) {
    std::vector<std::string> out;
    int depth = 0;
    std::string cur;
    for (char c : s) {
        if (c == '(') ++depth;
        if (c == ')') --depth;
        if (c == sep && depth == 0) {
            out.push_back(trim(cur));
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(trim(cur));
    return out;
}

const ConfigEntry* ConfigSection::find(const std::string& key) const {
    // Keys keep their spelling (constants are case-sensitive) but match case-insensitively.
    const std::string lk = toLower(key);
    for (const auto& e : entries)
        if (toLower(e.key) == lk) return &e;
    return nullptr;
}

Config Config::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config file '" + path + "'");
    Config cfg;
    cfg.path = path;
    cfg.sections.push_back(ConfigSection{"", 0, {}});
    std::string raw;
    int lineNo = 0;
    while (std::getline(in, raw)) {
        ++lineNo;
        std::string line = raw;
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        auto err = [&](const std::string& msg) {
            throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": " + msg);
        };
        if (line.front() == '[') {
            if (line.back() != ']') err("section header must end with ']'");
            std::string name = toLower(trim(line.substr(1, line.size() - 2)));
            if (name.empty()) err("empty section name");
            for (const auto& s : cfg.sections)
                if (s.name == name) err("duplicate section [" + name + "] (first defined on line " + std::to_string(s.line) + ")");
            cfg.sections.push_back(ConfigSection{name, lineNo, {}});
            continue;
        }
        size_t eq = line.find('=');
        if (eq == std::string::npos) err("expected 'key = value'");
        std::string key = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));
        if (key.empty()) err("missing key before '='");
        ConfigSection& sec = cfg.sections.back();
        if (sec.find(key)) err("duplicate key '" + key + "'");
        sec.entries.push_back(ConfigEntry{key, value, lineNo});
    }
    return cfg;
}

void Config::applyOverride(const std::string& spec) {
    size_t eq = spec.find('=');
    if (eq == std::string::npos) throw std::runtime_error("--set expects section.key=value, got '" + spec + "'");
    std::string lhs = trim(spec.substr(0, eq));
    std::string value = trim(spec.substr(eq + 1));
    size_t dot = lhs.rfind('.');
    std::string secName = dot == std::string::npos ? "" : toLower(lhs.substr(0, dot));
    std::string key = dot == std::string::npos ? lhs : lhs.substr(dot + 1);
    if (key.empty()) throw std::runtime_error("--set: missing key in '" + spec + "'");
    ConfigSection* sec = nullptr;
    for (auto& s : sections)
        if (s.name == secName) sec = &s;
    if (!sec) {
        sections.push_back(ConfigSection{secName, 0, {}});
        sec = &sections.back();
    }
    for (auto& e : sec->entries) {
        if (toLower(e.key) == toLower(key)) {
            e.value = value;
            e.line = 0;
            return;
        }
    }
    sec->entries.push_back(ConfigEntry{key, value, 0});
}

const ConfigSection* Config::section(const std::string& name) const {
    for (const auto& s : sections)
        if (s.name == name) return &s;
    return nullptr;
}

std::vector<const ConfigSection*> Config::sectionsWithPrefix(const std::string& prefix) const {
    std::vector<const ConfigSection*> out;
    std::string p = prefix + ".";
    for (const auto& s : sections)
        if (s.name.size() > p.size() && s.name.compare(0, p.size(), p) == 0) out.push_back(&s);
    return out;
}

int Config::warnUnused() const {
    int count = 0;
    for (const auto& s : sections) {
        for (const auto& e : s.entries) {
            if (e.used) continue;
            ++count;
            std::string loc = e.line > 0 ? path + ":" + std::to_string(e.line) : std::string("--set");
            std::fprintf(stderr, "warning: %s: unknown or unused key '%s' in [%s]\n", loc.c_str(), e.key.c_str(),
                         s.name.empty() ? "global" : s.name.c_str());
        }
    }
    return count;
}

// ---------------------------------------------------------------------------

SectionReader::SectionReader(const Config& cfg, const ConfigSection* sec, const Constants& constants)
    : cfg_(cfg), sec_(sec), constants_(constants) {}

const std::string& SectionReader::name() const {
    static const std::string empty;
    return sec_ ? sec_->name : empty;
}

const ConfigEntry* SectionReader::entry(const std::string& key) const {
    if (!sec_) return nullptr;
    const ConfigEntry* e = sec_->find(key);
    if (e) e->used = true;
    return e;
}

bool SectionReader::has(const std::string& key) const { return sec_ && sec_->find(key) != nullptr; }

std::string SectionReader::where(const std::string& key) const {
    const ConfigEntry* e = sec_ ? sec_->find(key) : nullptr;
    std::string loc = (e && e->line > 0) ? cfg_.path + ":" + std::to_string(e->line) : cfg_.path;
    std::string sec = sec_ ? (sec_->name.empty() ? "global" : sec_->name) : "?";
    return loc + ": [" + sec + "] " + key;
}

void SectionReader::fail(const std::string& key, const std::string& msg) const {
    throw std::runtime_error(where(key) + ": " + msg);
}

std::string SectionReader::str(const std::string& key, const std::string& def) const {
    const ConfigEntry* e = entry(key);
    return e ? e->value : def;
}

std::string SectionReader::strRequired(const std::string& key) const {
    const ConfigEntry* e = entry(key);
    if (!e) fail(key, "required key is missing");
    return e->value;
}

double SectionReader::number(const std::string& key, double def) const {
    const ConfigEntry* e = entry(key);
    if (!e) return def;
    try {
        Expr ex = Expr::parse(e->value, constants_);
        if (ex.usesSpace() || ex.usesT()) fail(key, "must be a constant (x, y, z, t not allowed here)");
        double v = ex.eval(0, 0, 0, 0);
        if (!std::isfinite(v)) fail(key, "value is not finite");
        return v;
    } catch (const std::runtime_error& err) {
        std::string m = err.what();
        if (m.rfind(cfg_.path, 0) == 0) throw;
        fail(key, m);
    }
}

double SectionReader::numberRequired(const std::string& key) const {
    if (!has(key)) {
        entry(key);
        fail(key, "required key is missing");
    }
    return number(key, 0.0);
}

int SectionReader::integer(const std::string& key, int def) const {
    double v = number(key, def);
    if (std::fabs(v - std::round(v)) > 1e-9) fail(key, "must be an integer");
    return static_cast<int>(std::lround(v));
}

bool SectionReader::boolean(const std::string& key, bool def) const {
    const ConfigEntry* e = entry(key);
    if (!e) return def;
    std::string v = toLower(e->value);
    if (v == "true" || v == "yes" || v == "on" || v == "1") return true;
    if (v == "false" || v == "no" || v == "off" || v == "0") return false;
    fail(key, "expected true/false, got '" + e->value + "'");
}

Expr SectionReader::expr(const std::string& key, const std::string& def, const char* allowed) const {
    const ConfigEntry* e = entry(key);
    std::string text = e ? e->value : def;
    if (text.empty()) return Expr();
    Expr ex;
    try {
        ex = Expr::parse(text, constants_);
    } catch (const std::runtime_error& err) {
        fail(key, err.what());
    }
    std::string a = allowed ? allowed : "";
    auto check = [&](bool used, char v) {
        if (used && a.find(v) == std::string::npos)
            fail(key, std::string("variable '") + v + "' is not allowed here (allowed: " +
                          (a.empty() ? std::string("none") : a) + ")");
    };
    check(ex.usesX(), 'x');
    check(ex.usesY(), 'y');
    check(ex.usesZ(), 'z');
    check(ex.usesT(), 't');
    return ex;
}

Expr SectionReader::exprRequired(const std::string& key, const char* allowed) const {
    if (!has(key)) {
        entry(key);
        fail(key, "required key is missing");
    }
    return expr(key, "", allowed);
}

std::vector<double> SectionReader::numbers(const std::string& key) const {
    const ConfigEntry* e = entry(key);
    std::vector<double> out;
    if (!e) return out;
    for (const std::string& part : splitTopLevel(e->value)) {
        if (part.empty()) fail(key, "empty list element");
        try {
            Expr ex = Expr::parse(part, constants_);
            if (ex.usesSpace() || ex.usesT()) fail(key, "list values must be constants");
            out.push_back(ex.eval(0, 0, 0, 0));
        } catch (const std::runtime_error& err) {
            std::string m = err.what();
            if (m.rfind(cfg_.path, 0) == 0) throw;
            fail(key, m);
        }
    }
    return out;
}

}  // namespace fc
