#include "expr.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace fc {

namespace {

enum Op : uint8_t {
    OP_PUSH, OP_X, OP_Y, OP_Z, OP_T,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD, OP_POW,
    OP_NEG, OP_NOT,
    OP_LT, OP_GT, OP_LE, OP_GE, OP_EQ, OP_NE, OP_AND, OP_OR,
    OP_FN1, OP_FN2, OP_FN3,
};

enum Fn : uint8_t {
    F_SIN, F_COS, F_TAN, F_ASIN, F_ACOS, F_ATAN, F_SINH, F_COSH, F_TANH, F_EXP, F_LOG, F_LOG10, F_LOG2,
    F_SQRT, F_ABS, F_FLOOR, F_CEIL, F_ROUND, F_SIGN, F_STEP,
    F_ATAN2, F_MIN, F_MAX, F_POW, F_MOD, F_HYPOT,
    F_CLAMP, F_IF,
};

struct FnInfo {
    const char* name;
    int arity;
    Fn fn;
};

const FnInfo kFunctions[] = {
    {"sin", 1, F_SIN},     {"cos", 1, F_COS},     {"tan", 1, F_TAN},     {"asin", 1, F_ASIN},
    {"acos", 1, F_ACOS},   {"atan", 1, F_ATAN},   {"sinh", 1, F_SINH},   {"cosh", 1, F_COSH},
    {"tanh", 1, F_TANH},   {"exp", 1, F_EXP},     {"log", 1, F_LOG},     {"ln", 1, F_LOG},
    {"log10", 1, F_LOG10}, {"log2", 1, F_LOG2},   {"sqrt", 1, F_SQRT},   {"abs", 1, F_ABS},
    {"floor", 1, F_FLOOR}, {"ceil", 1, F_CEIL},   {"round", 1, F_ROUND}, {"sign", 1, F_SIGN},
    {"step", 1, F_STEP},   {"atan2", 2, F_ATAN2}, {"min", 2, F_MIN},     {"max", 2, F_MAX},
    {"pow", 2, F_POW},     {"mod", 2, F_MOD},     {"hypot", 2, F_HYPOT}, {"clamp", 3, F_CLAMP},
    {"if", 3, F_IF},
};

constexpr int kMaxStack = 64;

double fn1(uint8_t f, double a) {
    switch (f) {
        case F_SIN: return std::sin(a);
        case F_COS: return std::cos(a);
        case F_TAN: return std::tan(a);
        case F_ASIN: return std::asin(a);
        case F_ACOS: return std::acos(a);
        case F_ATAN: return std::atan(a);
        case F_SINH: return std::sinh(a);
        case F_COSH: return std::cosh(a);
        case F_TANH: return std::tanh(a);
        case F_EXP: return std::exp(a);
        case F_LOG: return std::log(a);
        case F_LOG10: return std::log10(a);
        case F_LOG2: return std::log2(a);
        case F_SQRT: return std::sqrt(a);
        case F_ABS: return std::fabs(a);
        case F_FLOOR: return std::floor(a);
        case F_CEIL: return std::ceil(a);
        case F_ROUND: return std::round(a);
        case F_SIGN: return (a > 0) - (a < 0);
        case F_STEP: return a >= 0.0 ? 1.0 : 0.0;
        default: return 0.0;
    }
}

double fn2(uint8_t f, double a, double b) {
    switch (f) {
        case F_ATAN2: return std::atan2(a, b);
        case F_MIN: return std::fmin(a, b);
        case F_MAX: return std::fmax(a, b);
        case F_POW: return std::pow(a, b);
        case F_MOD: return std::fmod(a, b);
        case F_HYPOT: return std::hypot(a, b);
        default: return 0.0;
    }
}

double fn3(uint8_t f, double a, double b, double c) {
    switch (f) {
        case F_CLAMP: return std::fmin(std::fmax(a, b), c);
        case F_IF: return a != 0.0 ? b : c;
        default: return 0.0;
    }
}

}  // namespace

// Recursive-descent parser emitting postfix code.
//   or     := and ('||' and)*
//   and    := cmp ('&&' cmp)*
//   cmp    := add (('<'|'>'|'<='|'>='|'=='|'!=') add)*
//   add    := mul (('+'|'-') mul)*
//   mul    := unary (('*'|'/'|'%') unary)*
//   unary  := ('-'|'+'|'!') unary | pow
//   pow    := primary ('^' unary)?          (right associative, -2^2 = -4)
//   primary:= number | name | name '(' args ')' | '(' or ')'
class ExprParser {
public:
    ExprParser(const std::string& s, const std::map<std::string, double>& constants, Expr& out,
               const Expr::VarNames& names)
        : s_(s), constants_(constants), out_(out), names_(names) {}

    void run() {
        skipWs();
        if (pos_ >= s_.size()) fail("empty expression");
        parseOr();
        skipWs();
        if (pos_ != s_.size()) fail(std::string("unexpected '") + s_[pos_] + "'");
    }

private:
    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("in expression \"" + s_ + "\" at position " + std::to_string(pos_ + 1) + ": " + msg);
    }

    void skipWs() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }

    bool match(const char* tok) {
        skipWs();
        size_t len = std::char_traits<char>::length(tok);
        if (s_.compare(pos_, len, tok) == 0) {
            pos_ += len;
            return true;
        }
        return false;
    }

    // Matches a single-character operator that must not be the prefix of a longer one.
    bool matchSingle(char c, const char* notFollowedBy) {
        skipWs();
        if (pos_ < s_.size() && s_[pos_] == c) {
            if (pos_ + 1 < s_.size() && notFollowedBy && std::char_traits<char>::find(notFollowedBy, std::char_traits<char>::length(notFollowedBy), s_[pos_ + 1]))
                return false;
            ++pos_;
            return true;
        }
        return false;
    }

    void emit(uint8_t op, uint8_t fn = 0, double v = 0.0) {
        out_.code_.push_back({op, fn, v});
        switch (op) {
            case OP_PUSH: case OP_X: case OP_Y: case OP_Z: case OP_T: depth_ += 1; break;
            case OP_NEG: case OP_NOT: case OP_FN1: break;
            case OP_FN3: depth_ -= 2; break;
            default: depth_ -= 1; break;  // binary operators and FN2
        }
        if (depth_ > kMaxStack) fail("expression too deeply nested");
    }

    void parseOr() {
        parseAnd();
        while (match("||")) {
            parseAnd();
            emit(OP_OR);
        }
    }

    void parseAnd() {
        parseCmp();
        while (match("&&")) {
            parseCmp();
            emit(OP_AND);
        }
    }

    void parseCmp() {
        parseAdd();
        for (;;) {
            uint8_t op;
            if (match("<=")) op = OP_LE;
            else if (match(">=")) op = OP_GE;
            else if (match("==")) op = OP_EQ;
            else if (match("!=")) op = OP_NE;
            else if (matchSingle('<', nullptr)) op = OP_LT;
            else if (matchSingle('>', nullptr)) op = OP_GT;
            else break;
            parseAdd();
            emit(op);
        }
    }

    void parseAdd() {
        parseMul();
        for (;;) {
            if (matchSingle('+', nullptr)) {
                parseMul();
                emit(OP_ADD);
            } else if (matchSingle('-', nullptr)) {
                parseMul();
                emit(OP_SUB);
            } else {
                break;
            }
        }
    }

    void parseMul() {
        parseUnary();
        for (;;) {
            if (matchSingle('*', nullptr)) {
                parseUnary();
                emit(OP_MUL);
            } else if (matchSingle('/', nullptr)) {
                parseUnary();
                emit(OP_DIV);
            } else if (matchSingle('%', nullptr)) {
                parseUnary();
                emit(OP_MOD);
            } else {
                break;
            }
        }
    }

    void parseUnary() {
        if (matchSingle('-', nullptr)) {
            parseUnary();
            emit(OP_NEG);
        } else if (matchSingle('+', nullptr)) {
            parseUnary();
        } else if (matchSingle('!', "=")) {
            parseUnary();
            emit(OP_NOT);
        } else {
            parsePow();
        }
    }

    void parsePow() {
        parsePrimary();
        if (matchSingle('^', nullptr)) {
            parseUnary();
            emit(OP_POW);
        }
    }

    void parsePrimary() {
        skipWs();
        if (pos_ >= s_.size()) fail("unexpected end of expression");
        char c = s_[pos_];
        if (std::isdigit(static_cast<unsigned char>(c)) ||
            (c == '.' && pos_ + 1 < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_ + 1])))) {
            const char* begin = s_.c_str() + pos_;
            char* end = nullptr;
            double v = std::strtod(begin, &end);
            pos_ += static_cast<size_t>(end - begin);
            emit(OP_PUSH, 0, v);
            return;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t start = pos_;
            while (pos_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_')) ++pos_;
            std::string name = s_.substr(start, pos_ - start);
            skipWs();
            if (pos_ < s_.size() && s_[pos_] == '(') {
                ++pos_;
                const FnInfo* info = nullptr;
                for (const auto& f : kFunctions)
                    if (name == f.name) info = &f;
                if (!info) fail("unknown function '" + name + "'");
                int args = 0;
                skipWs();
                if (!match(")")) {
                    for (;;) {
                        parseOr();
                        ++args;
                        if (match(")")) break;
                        if (!match(",")) fail("expected ',' or ')' in call to " + name);
                    }
                }
                if (args != info->arity)
                    fail("function '" + name + "' takes " + std::to_string(info->arity) + " argument(s), got " +
                         std::to_string(args));
                emit(info->arity == 1 ? OP_FN1 : info->arity == 2 ? OP_FN2 : OP_FN3, info->fn);
                return;
            }
            for (int slot = 0; slot < 4; ++slot) {
                if (names_[slot].empty() || name != names_[slot]) continue;
                emit(static_cast<uint8_t>(OP_X + slot));
                bool* uses[4] = {&out_.usesX_, &out_.usesY_, &out_.usesZ_, &out_.usesT_};
                *uses[slot] = true;
                return;
            }
            if (name == "pi") {
                emit(OP_PUSH, 0, 3.14159265358979323846);
            } else if (name == "e") {
                emit(OP_PUSH, 0, 2.71828182845904523536);
            } else {
                auto it = constants_.find(name);
                if (it == constants_.end()) fail("unknown name '" + name + "'");
                emit(OP_PUSH, 0, it->second);
            }
            return;
        }
        if (c == '(') {
            ++pos_;
            parseOr();
            if (!match(")")) fail("missing ')'");
            return;
        }
        fail(std::string("unexpected '") + c + "'");
    }

    const std::string& s_;
    const std::map<std::string, double>& constants_;
    Expr& out_;
    Expr::VarNames names_;
    size_t pos_ = 0;
    int depth_ = 0;
};

Expr Expr::parse(const std::string& text, const std::map<std::string, double>& constants, const VarNames& names) {
    Expr e;
    e.text_ = text;
    ExprParser p(text, constants, e, names);
    p.run();
    return e;
}

Expr Expr::constant(double value) {
    Expr e;
    e.code_.push_back({OP_PUSH, 0, value});
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", value);
    e.text_ = buf;
    return e;
}

double Expr::eval(double x, double y, double z, double t) const {
    double st[kMaxStack + 1];
    int sp = 0;
    for (const Ins& in : code_) {
        switch (in.op) {
            case OP_PUSH: st[sp++] = in.value; break;
            case OP_X: st[sp++] = x; break;
            case OP_Y: st[sp++] = y; break;
            case OP_Z: st[sp++] = z; break;
            case OP_T: st[sp++] = t; break;
            case OP_ADD: --sp; st[sp - 1] += st[sp]; break;
            case OP_SUB: --sp; st[sp - 1] -= st[sp]; break;
            case OP_MUL: --sp; st[sp - 1] *= st[sp]; break;
            case OP_DIV: --sp; st[sp - 1] /= st[sp]; break;
            case OP_MOD: --sp; st[sp - 1] = std::fmod(st[sp - 1], st[sp]); break;
            case OP_POW: --sp; st[sp - 1] = std::pow(st[sp - 1], st[sp]); break;
            case OP_NEG: st[sp - 1] = -st[sp - 1]; break;
            case OP_NOT: st[sp - 1] = st[sp - 1] == 0.0 ? 1.0 : 0.0; break;
            case OP_LT: --sp; st[sp - 1] = st[sp - 1] < st[sp]; break;
            case OP_GT: --sp; st[sp - 1] = st[sp - 1] > st[sp]; break;
            case OP_LE: --sp; st[sp - 1] = st[sp - 1] <= st[sp]; break;
            case OP_GE: --sp; st[sp - 1] = st[sp - 1] >= st[sp]; break;
            case OP_EQ: --sp; st[sp - 1] = st[sp - 1] == st[sp]; break;
            case OP_NE: --sp; st[sp - 1] = st[sp - 1] != st[sp]; break;
            case OP_AND: --sp; st[sp - 1] = (st[sp - 1] != 0.0) && (st[sp] != 0.0); break;
            case OP_OR: --sp; st[sp - 1] = (st[sp - 1] != 0.0) || (st[sp] != 0.0); break;
            case OP_FN1: st[sp - 1] = fn1(in.fn, st[sp - 1]); break;
            case OP_FN2: --sp; st[sp - 1] = fn2(in.fn, st[sp - 1], st[sp]); break;
            case OP_FN3: sp -= 2; st[sp - 1] = fn3(in.fn, st[sp - 1], st[sp], st[sp + 1]); break;
        }
    }
    return sp > 0 ? st[0] : 0.0;
}

}  // namespace fc
