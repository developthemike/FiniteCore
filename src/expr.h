#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace fc {

// A small math-expression evaluator used for every value in the config file.
//
//   variables : x, y, z (position, m; z = 0 in 2D), t (time, s)
//   constants : pi, e, and everything defined in the [constants] section
//   operators : + - * / % ^  < > <= >= == !=  && || !   (comparisons yield 0/1)
//   functions : sin cos tan asin acos atan sinh cosh tanh exp log log10 log2
//               sqrt abs floor ceil round sign step(x>=0)
//               atan2 min max pow mod hypot   clamp(v,lo,hi)  if(c,a,b)
//
// Expressions are compiled once to a compact stack program.
class Expr {
public:
    // Names of the four variable slots that eval() fills; "" = slot unused.
    using VarNames = std::array<std::string, 4>;

    Expr() = default;
    static Expr parse(const std::string& text, const std::map<std::string, double>& constants,
                      const VarNames& names = {"x", "y", "z", "t"});
    static Expr constant(double value);

    double eval(double x, double y, double z, double t) const;

    bool valid() const { return !code_.empty(); }
    bool usesX() const { return usesX_; }
    bool usesY() const { return usesY_; }
    bool usesZ() const { return usesZ_; }
    bool usesT() const { return usesT_; }
    bool usesSpace() const { return usesX_ || usesY_ || usesZ_; }
    const std::string& text() const { return text_; }

private:
    friend class ExprParser;
    struct Ins {
        uint8_t op;
        uint8_t fn;
        double value;
    };
    std::vector<Ins> code_;
    std::string text_;
    bool usesX_ = false, usesY_ = false, usesZ_ = false, usesT_ = false;
};

}  // namespace fc
