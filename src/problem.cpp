#include "problem.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace fc {

namespace {

const char* kColormaps[] = {"inferno", "viridis", "turbo", "gray"};
const char* kViewModes[] = {"surface", "clip", "octant", "slices"};

std::string fmt(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", v);
    return buf;
}

Region parseRegion(const SectionReader& r, bool required, int dim) {
    const char* keys[] = {"rect", "circle", "box", "sphere", "where"};
    int count = 0;
    for (const char* k : keys) count += r.has(k) ? 1 : 0;
    if (count > 1) r.fail("rect", "use only one of rect, circle, box, sphere or where");
    Region g;
    auto needs3d = [&](const char* key) {
        if (dim != 3) r.fail(key, std::string(key) + " regions need a 3D (box) mesh; use rect or circle in 2D");
    };
    if (r.has("rect")) {
        std::vector<double> v = r.numbers("rect");
        if (v.size() != 4) r.fail("rect", "expected rect = x0, y0, x1, y1");
        g.kind = Region::Kind::Rect;
        g.v[0] = std::fmin(v[0], v[2]);
        g.v[1] = std::fmin(v[1], v[3]);
        g.v[2] = std::fmax(v[0], v[2]);
        g.v[3] = std::fmax(v[1], v[3]);
    } else if (r.has("circle")) {
        std::vector<double> v = r.numbers("circle");
        if (v.size() != 3 || !(v[2] > 0)) r.fail("circle", "expected circle = cx, cy, radius (radius > 0)");
        g.kind = Region::Kind::Circle;
        std::copy(v.begin(), v.end(), g.v);
    } else if (r.has("box")) {
        needs3d("box");
        std::vector<double> v = r.numbers("box");
        if (v.size() != 6) r.fail("box", "expected box = x0, y0, z0, x1, y1, z1");
        g.kind = Region::Kind::Box;
        for (int a = 0; a < 3; ++a) {
            g.v[a] = std::fmin(v[a], v[a + 3]);
            g.v[a + 3] = std::fmax(v[a], v[a + 3]);
        }
    } else if (r.has("sphere")) {
        needs3d("sphere");
        std::vector<double> v = r.numbers("sphere");
        if (v.size() != 4 || !(v[3] > 0)) r.fail("sphere", "expected sphere = cx, cy, cz, radius (radius > 0)");
        g.kind = Region::Kind::Sphere;
        std::copy(v.begin(), v.end(), g.v);
    } else if (r.has("where")) {
        g.kind = Region::Kind::Where;
        g.where = r.expr("where", "", dim == 3 ? "xyz" : "xy");
    } else if (required) {
        r.fail("rect", dim == 3 ? "a region is required: box, sphere, rect, circle or where"
                                : "a region is required: rect = x0, y0, x1, y1 | circle = cx, cy, r | where = <expression>");
    }
    return g;
}

double positive(const SectionReader& r, const std::string& key, double v) {
    if (!(v > 0)) r.fail(key, "must be > 0");
    return v;
}

int countFromList(const SectionReader& r, const std::string& key, double v) {
    if (std::fabs(v - std::round(v)) > 1e-9 || v < 1) r.fail(key, "cell counts must be positive integers");
    return static_cast<int>(std::lround(v));
}

// Rectangle (2D) or box (3D) mesh parameters.
void parseBlock(const SectionReader& r, const std::string& mtype, MeshDef& md) {
    const std::vector<double> size = r.numbers("size");
    if (mtype == "box") {
        md.dim = 3;
    } else if (mtype == "rectangle") {
        md.dim = 2;
    } else if (mtype.empty()) {
        md.dim = size.size() == 3 ? 3 : 2;
    } else {
        r.fail("type", "expected rectangle (2D), box or tube (3D)");
    }
    const int dim = md.dim;
    md.kind = dim == 3 ? "box" : "rectangle";
    const bool hasWH = r.has("width") || r.has("height");
    if (!size.empty()) {
        if (static_cast<int>(size.size()) != dim)
            r.fail("size", dim == 3 ? "expected size = Lx, Ly, Lz" : "expected size = Lx, Ly");
        if (hasWH) r.fail("size", "use either size or width/height");
        for (int a = 0; a < dim; ++a) md.size[a] = positive(r, "size", size[a]);
    } else if (dim == 2) {
        md.size[0] = positive(r, "width", r.numberRequired("width"));
        md.size[1] = positive(r, "height", r.numberRequired("height"));
    } else {
        r.fail("size", "a box mesh needs size = Lx, Ly, Lz");
    }
    const std::vector<double> origin = r.numbers("origin");
    if (!origin.empty()) {
        if (static_cast<int>(origin.size()) != dim)
            r.fail("origin", dim == 3 ? "expected origin = x0, y0, z0" : "expected origin = x0, y0");
        for (int a = 0; a < dim; ++a) md.origin[a] = origin[a];
    }
    const std::vector<double> cellList = r.numbers("cells");
    if (!cellList.empty()) {
        if (static_cast<int>(cellList.size()) != dim)
            r.fail("cells", dim == 3 ? "expected cells = nx, ny, nz" : "expected cells = nx, ny");
        if (r.has("nx") || r.has("ny") || r.has("nz")) r.fail("cells", "use either cells or nx/ny/nz");
        for (int a = 0; a < dim; ++a) md.cells[a] = countFromList(r, "cells", cellList[a]);
    } else {
        // Missing counts keep the cells roughly cubic.
        const int nx = r.integer("nx", dim == 3 ? 32 : 64);
        md.cells[0] = nx;
        for (int a = 1; a < dim; ++a) {
            const int def = std::max(1, static_cast<int>(std::lround(nx * md.size[a] / md.size[0])));
            md.cells[a] = r.integer(a == 1 ? "ny" : "nz", def);
        }
    }
    for (int a = 0; a < dim; ++a)
        if (md.cells[a] < 1) r.fail("nx", "cell counts must be >= 1");
    if (dim == 2) {
        md.pattern = toLower(r.str("pattern", "alternating"));
        if (md.pattern != "alternating" && md.pattern != "right" && md.pattern != "left")
            r.fail("pattern", "expected alternating, right or left");
        md.thickness = positive(r, "thickness", r.number("thickness", 1.0));
    } else {
        if (r.has("thickness")) r.fail("thickness", "only for 2D plate models (a box is meshed through its depth)");
        if (r.has("pattern")) r.fail("pattern", "only for 2D meshes");
    }
}

// Tube: a solid swept along a circular arc with a radius profile radius(s)
// (s in [0, 1] along the centre line) and a cross-section shape
// section(angle, s); optionally hollow with a round hole of radius cavity(s).
void parseTube(const SectionReader& r, const Constants& consts, MeshDef& md) {
    md.kind = "tube";
    md.dim = 3;
    md.length = positive(r, "length", r.numberRequired("length"));
    md.bend = r.number("bend", 0.0);
    if (md.bend < 0) r.fail("bend", "must be >= 0 (0 = straight)");
    auto profile = [&](const std::string& key, const std::string& def, const Expr::VarNames& vars) {
        const std::string text = def.empty() ? r.strRequired(key) : r.str(key, def);
        try {
            Expr e = Expr::parse(text, consts, vars);
            return e;
        } catch (const std::runtime_error& err) {
            r.fail(key, err.what());
        }
    };
    constexpr double pi = 3.14159265358979323846;
    md.radius = profile("radius", "", {"s", "", "", ""});
    md.section = profile("section", "1", {"angle", "s", "", ""});
    if (r.has("cavity")) md.cavity = profile("cavity", "", {"s", "", "", ""});
    for (int k = 0; k <= 1000; ++k) {
        const double sv = k / 1000.0;
        const double rv = md.radius.eval(sv, 0, 0, 0);
        if (!(rv > 0) || !std::isfinite(rv))
            r.fail("radius", "must be positive for every s in [0, 1] (radius(" + fmt(sv) + ") = " + fmt(rv) + ")");
        if (md.cavity.valid()) {
            const double cv = md.cavity.eval(sv, 0, 0, 0);
            if (!(cv > 0) || !std::isfinite(cv))
                r.fail("cavity", "must be positive for every s in [0, 1]: the hole runs through the tube; leave "
                                 "cavity out for a solid tube (cavity(" + fmt(sv) + ") = " + fmt(cv) + ")");
        }
    }
    for (int i = 0; i <= 100; ++i) {
        const double sv = i / 100.0;
        for (int k = 0; k <= 360; ++k) {
            const double av = -pi + 2.0 * pi * k / 360.0;
            const double sec = md.section.eval(av, sv, 0, 0);
            if (!(sec > 0) || !std::isfinite(sec))
                r.fail("section", "must be positive for every angle and s (section(" + fmt(av) + ", " + fmt(sv) +
                                      ") = " + fmt(sec) + ")");
        }
    }
    const std::vector<double> cells = r.numbers("cells");
    if (!cells.empty()) {
        if (cells.size() != 2) r.fail("cells", "a tube expects cells = along, across");
        md.cells[0] = countFromList(r, "cells", cells[0]);
        md.cells[1] = countFromList(r, "cells", cells[1]);
    } else {
        md.cells[0] = 64;
        md.cells[1] = 12;
    }
    if (md.cells[1] < 2) r.fail("cells", "a tube needs at least 2 cells across");
    const std::vector<double> origin = r.numbers("origin");
    if (!origin.empty()) {
        if (origin.size() != 3) r.fail("origin", "expected origin = x0, y0, z0 (middle of the centre line)");
        for (int a = 0; a < 3; ++a) md.origin[a] = origin[a];
    }
    for (const char* key : {"size", "width", "height", "nx", "ny", "nz", "thickness", "pattern"})
        if (r.has(key)) r.fail(key, "not used by tube meshes (use length, bend, radius, section, cavity, cells)");
}

}  // namespace

const char* colormapName(int index) {
    if (index < 0 || index > 3) index = 0;
    return kColormaps[index];
}

const char* viewModeName(int mode) {
    if (mode < 0 || mode > 3) mode = 0;
    return kViewModes[mode];
}

const char* bcTypeName(BcType t) {
    switch (t) {
        case BcType::Dirichlet: return "dirichlet";
        case BcType::Neumann: return "neumann";
        case BcType::Robin: return "robin";
        default: return "insulated";
    }
}

bool Region::contains(double x, double y, double z) const {
    switch (kind) {
        case Kind::Rect: return x >= v[0] && x <= v[2] && y >= v[1] && y <= v[3];
        case Kind::Circle: {
            const double dx = x - v[0], dy = y - v[1];
            return dx * dx + dy * dy <= v[2] * v[2];
        }
        case Kind::Box: return x >= v[0] && x <= v[3] && y >= v[1] && y <= v[4] && z >= v[2] && z <= v[5];
        case Kind::Sphere: {
            const double dx = x - v[0], dy = y - v[1], dz = z - v[2];
            return dx * dx + dy * dy + dz * dz <= v[3] * v[3];
        }
        case Kind::Where: return where.eval(x, y, z, 0.0) != 0.0;
        default: return true;
    }
}

std::string Region::describe() const {
    switch (kind) {
        case Kind::Rect: return "rect (" + fmt(v[0]) + ", " + fmt(v[1]) + ")-(" + fmt(v[2]) + ", " + fmt(v[3]) + ")";
        case Kind::Circle: return "circle at (" + fmt(v[0]) + ", " + fmt(v[1]) + ") r=" + fmt(v[2]);
        case Kind::Box:
            return "box (" + fmt(v[0]) + ", " + fmt(v[1]) + ", " + fmt(v[2]) + ")-(" + fmt(v[3]) + ", " + fmt(v[4]) +
                   ", " + fmt(v[5]) + ")";
        case Kind::Sphere:
            return "sphere at (" + fmt(v[0]) + ", " + fmt(v[1]) + ", " + fmt(v[2]) + ") r=" + fmt(v[3]);
        case Kind::Where: return "where " + where.text();
        default: return "everywhere";
    }
}

ProblemDef parseProblem(const Config& cfg) {
    ProblemDef p;
    p.configPath = cfg.path;
    const std::string stem = std::filesystem::path(cfg.path).stem().string();

    // [constants]: evaluated in order, each may use the previous ones.
    Constants consts;
    if (const ConfigSection* cs = cfg.section("constants")) {
        for (const ConfigEntry& e : cs->entries) {
            e.used = true;
            const std::string loc = (e.line > 0 ? cfg.path + ":" + std::to_string(e.line) : std::string("--set")) +
                                    ": [constants] " + e.key + ": ";
            if (e.key == "x" || e.key == "y" || e.key == "z" || e.key == "t" || e.key == "pi" || e.key == "e")
                throw std::runtime_error(loc + "'" + e.key + "' is a reserved name");
            try {
                Expr ex = Expr::parse(e.value, consts);
                if (ex.usesSpace() || ex.usesT()) throw std::runtime_error("constants cannot use x, y, z or t");
                consts[e.key] = ex.eval(0, 0, 0, 0);
            } catch (const std::runtime_error& err) {
                throw std::runtime_error(loc + err.what());
            }
        }
    }
    auto reader = [&](const std::string& name) { return SectionReader(cfg, cfg.section(name), consts); };

    SectionReader global = reader("");
    p.title = global.str("title", stem);

    // ---- mesh -------------------------------------------------------------
    SectionReader mesh = reader("mesh");
    if (!mesh.exists()) throw std::runtime_error(cfg.path + ": missing [mesh] section");
    const std::string mtype = toLower(mesh.str("type", ""));
    if (mtype == "tube") {
        parseTube(mesh, consts, p.mesh);
    } else {
        parseBlock(mesh, mtype, p.mesh);
    }
    const int dim = p.mesh.dim;
    const std::string sp = dim == 3 ? "xyz" : "xy";
    const std::string spt = sp + "t";

    // ---- materials --------------------------------------------------------
    SectionReader mat = reader("material");
    if (!mat.exists()) throw std::runtime_error(cfg.path + ": missing [material] section");
    p.baseMaterial.name = "base";
    p.baseMaterial.conductivity = positive(mat, "conductivity", mat.numberRequired("conductivity"));
    p.baseMaterial.density = positive(mat, "density", mat.numberRequired("density"));
    p.baseMaterial.specificHeat = positive(mat, "specific_heat", mat.numberRequired("specific_heat"));
    for (const ConfigSection* s : cfg.sectionsWithPrefix("material")) {
        SectionReader r(cfg, s, consts);
        MaterialDef m;
        m.name = s->name.substr(std::string("material.").size());
        m.region = parseRegion(r, true, dim);
        m.conductivity = positive(r, "conductivity", r.number("conductivity", p.baseMaterial.conductivity));
        m.density = positive(r, "density", r.number("density", p.baseMaterial.density));
        m.specificHeat = positive(r, "specific_heat", r.number("specific_heat", p.baseMaterial.specificHeat));
        p.materials.push_back(m);
    }

    // ---- initial condition ------------------------------------------------
    SectionReader init = reader("initial");
    if (!init.exists()) throw std::runtime_error(cfg.path + ": missing [initial] section");
    p.initial = init.exprRequired("temperature", sp.c_str());

    // ---- boundary conditions ----------------------------------------------
    for (const ConfigSection* s : cfg.sectionsWithPrefix("boundary")) {
        SectionReader r(cfg, s, consts);
        BoundaryDef b;
        b.name = s->name.substr(std::string("boundary.").size());
        for (const std::string& part : splitTopLevel(r.str("side", b.name))) {
            if (part.empty()) r.fail("side", "empty side name");
            b.sides.push_back(toLower(part));
        }
        b.where = r.expr("where", "", sp.c_str());
        const std::string type = toLower(r.strRequired("type"));
        if (type == "dirichlet" || type == "temperature" || type == "fixed") {
            b.type = BcType::Dirichlet;
            b.temperature = r.exprRequired("temperature", sp.c_str());
        } else if (type == "neumann" || type == "flux") {
            b.type = BcType::Neumann;
            b.flux = r.exprRequired("flux", spt.c_str());
        } else if (type == "robin" || type == "convection") {
            b.type = BcType::Robin;
            b.h = r.exprRequired("h", sp.c_str());
            b.ambient = r.exprRequired("ambient", spt.c_str());
        } else if (type == "insulated" || type == "adiabatic") {
            b.type = BcType::Insulated;
        } else {
            r.fail("type", "expected dirichlet, neumann, robin or insulated");
        }
        p.boundaries.push_back(b);
    }

    // ---- heat sources -----------------------------------------------------
    for (const ConfigSection* s : cfg.sectionsWithPrefix("source")) {
        SectionReader r(cfg, s, consts);
        SourceDef src;
        src.name = s->name.substr(std::string("source.").size());
        src.region = parseRegion(r, false, dim);
        const bool hasPower = r.has("power"), hasDensity = r.has("power_density");
        if (hasPower == hasDensity) r.fail("power", "specify exactly one of power [W] or power_density [W/m^3]");
        if (hasPower) {
            src.total = true;
            src.value = r.expr("power", "", "t");
        } else {
            src.value = r.expr("power_density", "", spt.c_str());
        }
        p.sources.push_back(src);
    }

    // ---- plate faces (2D plate model) ---------------------------------------
    SectionReader faces = reader("faces");
    if (dim == 3) {
        if (faces.exists())
            throw std::runtime_error(cfg.path + ": [faces] is the 2D plate model; in 3D put robin boundaries on the "
                                                "faces instead (e.g. [boundary.air] side = top, bottom)");
    } else {
        p.faceH = faces.number("h", 0.0);
        if (p.faceH < 0) faces.fail("h", "must be >= 0");
        if (p.faceH > 0 && !faces.has("ambient")) faces.fail("ambient", "required when h > 0");
        p.faceAmbient = faces.expr("ambient", "0", spt.c_str());
        p.faceCount = faces.integer("count", 2);
        if (p.faceCount != 1 && p.faceCount != 2) faces.fail("count", "must be 1 or 2");
    }

    // ---- verification -----------------------------------------------------
    SectionReader ver = reader("verify");
    if (ver.has("exact")) {
        p.hasExact = true;
        p.exact = ver.expr("exact", "", spt.c_str());
    }

    // ---- time integration -------------------------------------------------
    SectionReader tm = reader("time");
    if (!tm.exists()) throw std::runtime_error(cfg.path + ": missing [time] section");
    p.time.end = positive(tm, "end", tm.numberRequired("end"));
    if (toLower(tm.str("dt", "auto")) == "auto") {
        p.time.dt = 0.0;
    } else {
        p.time.dt = positive(tm, "dt", tm.number("dt", 0.0));
    }
    p.time.scheme = toLower(tm.str("scheme", "backward_euler"));
    if (p.time.scheme == "explicit" || p.time.scheme == "forward_euler") {
        p.time.scheme = "explicit";
        p.time.theta = 0.0;
    } else if (p.time.scheme == "backward_euler" || p.time.scheme == "implicit") {
        p.time.scheme = "backward_euler";
        p.time.theta = 1.0;
    } else if (p.time.scheme == "crank_nicolson" || p.time.scheme == "cn") {
        p.time.scheme = "crank_nicolson";
        p.time.theta = 0.5;
    } else if (p.time.scheme == "theta") {
        p.time.theta = tm.numberRequired("theta");
        if (p.time.theta < 0 || p.time.theta > 1) tm.fail("theta", "must be in [0, 1]");
    } else {
        tm.fail("scheme", "expected explicit, backward_euler, crank_nicolson or theta");
    }
    p.time.mass = toLower(tm.str("mass", "auto"));
    if (p.time.mass != "auto" && p.time.mass != "consistent" && p.time.mass != "lumped")
        tm.fail("mass", "expected auto, consistent or lumped");

    // ---- linear solver ----------------------------------------------------
    SectionReader sv = reader("solver");
    p.solver.tolerance = sv.number("tolerance", 1e-6);
    if (!(p.solver.tolerance > 0 && p.solver.tolerance < 1)) sv.fail("tolerance", "must be in (0, 1)");
    p.solver.maxIterations = sv.integer("max_iterations", 1000);
    if (p.solver.maxIterations < 1) sv.fail("max_iterations", "must be >= 1");

    // ---- live view ----------------------------------------------------------
    SectionReader vw = reader("view");
    p.view.window = vw.boolean("window", true);
    p.view.width = toLower(vw.str("width", "auto")) == "auto" ? 0 : vw.integer("width", 0);
    p.view.height = vw.integer("height", 800);
    if (p.view.width < 0 || p.view.height < 120) vw.fail("height", "window size too small");
    p.view.playback = positive(vw, "playback", vw.number("playback", 12.0));
    const std::string range = toLower(vw.str("range", "auto"));
    if (range == "auto" || range == "dynamic") {
        p.view.range = range;
    } else {
        std::vector<double> rv = vw.numbers("range");
        if (rv.size() != 2 || !(rv[1] > rv[0])) vw.fail("range", "expected auto, dynamic or 'min, max' with max > min");
        p.view.range = "fixed";
        p.view.rangeMin = rv[0];
        p.view.rangeMax = rv[1];
    }
    const std::string cmap = toLower(vw.str("colormap", "inferno"));
    p.view.colormap = -1;
    for (int i = 0; i < 4; ++i)
        if (cmap == kColormaps[i]) p.view.colormap = i;
    if (p.view.colormap < 0) vw.fail("colormap", "expected inferno, viridis, turbo or gray");
    p.view.contours = vw.integer("contours", 10);
    if (p.view.contours < 0 || p.view.contours > 200) vw.fail("contours", "must be in [0, 200]");
    p.view.mesh = vw.boolean("mesh", false);
    p.view.unit = vw.str("unit", p.view.unit);
    if (dim == 3) {
        const std::string mode = toLower(vw.str("mode", "octant"));
        p.view.mode = -1;
        for (int i = 0; i < 4; ++i)
            if (mode == kViewModes[i]) p.view.mode = i;
        if (p.view.mode < 0) vw.fail("mode", "expected surface, clip, octant or slices");
        const std::string axis = toLower(vw.str("axis", "x"));
        if (axis != "x" && axis != "y" && axis != "z") vw.fail("axis", "expected x, y or z");
        p.view.axis = axis[0] - 'x';
        const std::vector<double> cut = vw.numbers("cut");
        if (!cut.empty()) {
            if (cut.size() != 3) vw.fail("cut", "expected cut = x, y, z");
            p.view.cutSet = true;
            std::copy(cut.begin(), cut.end(), p.view.cut);
        }
        p.view.yaw = vw.number("yaw", -60.0);
        p.view.pitch = std::clamp(vw.number("pitch", 25.0), -89.0, 89.0);
        p.view.zoom = positive(vw, "zoom", vw.number("zoom", 1.0));
        p.view.spin = vw.number("spin", 0.0);
    }

    // ---- file output --------------------------------------------------------
    SectionReader out = reader("output");
    p.output.directory = out.str("directory", "output/" + stem);
    if (toLower(out.str("interval", "auto")) == "auto") {
        p.output.interval = p.time.end / 120.0;
    } else {
        p.output.interval = positive(out, "interval", out.number("interval", 0.0));
    }
    p.output.png = out.boolean("png", false);
    p.output.gif = out.boolean("gif", true);
    p.output.vtk = out.boolean("vtk", false);
    p.output.csv = out.boolean("csv", true);
    p.output.width = toLower(out.str("width", "auto")) == "auto" ? 0 : out.integer("width", 0);
    p.output.height = out.integer("height", 600);
    if (p.output.width < 0 || p.output.height < 120 || p.output.height > 4096 || p.output.width > 4096)
        out.fail("height", "image size must be within 120..4096 pixels");
    p.output.gifFps = positive(out, "gif_fps", out.number("gif_fps", 20.0));
    if (p.output.gifFps > 50) out.fail("gif_fps", "GIF frame rate must be <= 50");

    SectionReader pr = reader("probes");
    if (const ConfigSection* ps = cfg.section("probes")) {
        for (const ConfigEntry& e : ps->entries) {
            std::vector<double> v = pr.numbers(e.key);
            if (static_cast<int>(v.size()) != dim) pr.fail(e.key, dim == 3 ? "expected name = x, y, z" : "expected name = x, y");
            p.output.probes.push_back({e.key, v[0], v[1], dim == 3 ? v[2] : 0.0});
        }
    }
    return p;
}

}  // namespace fc
