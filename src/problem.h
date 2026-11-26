#pragma once

#include <string>
#include <vector>

#include "config.h"
#include "expr.h"

namespace fc {

// Region of the domain, tested at element centroids (materials, sources).
// rect/circle use x and y only (in 3D they extend through all z).
struct Region {
    enum class Kind { All, Rect, Circle, Box, Sphere, Where } kind = Kind::All;
    double v[6] = {0, 0, 0, 0, 0, 0};
    Expr where;
    bool contains(double x, double y, double z) const;
    std::string describe() const;
};

struct MaterialDef {
    std::string name;
    Region region;
    double conductivity = 0;  // k   [W/(m K)]
    double density = 0;       // rho [kg/m^3]
    double specificHeat = 0;  // cp  [J/(kg K)]
};

enum class BcType { Insulated, Dirichlet, Neumann, Robin };
const char* bcTypeName(BcType t);

struct BoundaryDef {
    std::string name;
    std::vector<std::string> sides;  // mesh side names; "all" = every side
    Expr where;                      // optional filter on the facet centroid
    BcType type = BcType::Insulated;
    Expr temperature;  // Dirichlet: T(x, y, z)              [K or degC]
    Expr flux;         // Neumann: q(x, y, z, t), into body  [W/m^2]
    Expr h;            // Robin: h(x, y, z)                  [W/(m^2 K)]
    Expr ambient;      // Robin: T_inf(x, y, z, t)
};

struct SourceDef {
    std::string name;
    Region region;
    bool total = false;  // true: `power` [W] spread over the region; false: `power_density` [W/m^3]
    Expr value;
};

struct ProbeDef {
    std::string name;
    double x = 0, y = 0, z = 0;
};

struct MeshDef {
    std::string kind = "rectangle";  // rectangle (2D) | box | tube (3D)
    int dim = 2;
    double origin[3] = {0, 0, 0};
    double size[3] = {1, 1, 0};
    int cells[3] = {64, 64, 0};           // tube: along, across
    std::string pattern = "alternating";  // 2D diagonal pattern
    double thickness = 1.0;               // 2D plate thickness [m]; 1 = per unit depth
    // tube
    double length = 0.0, bend = 0.0;  // centre-line length and arc radius [m]
    Expr radius;   // radius(s) [m], s in [0, 1] along the tube
    Expr section;  // relative radius vs. angle around the centre line
};

struct TimeDef {
    double end = 1.0;
    double dt = 0.0;  // 0 = auto
    double theta = 1.0;
    std::string scheme = "backward_euler";
    std::string mass = "auto";  // auto | consistent | lumped
};

struct SolverDef {
    double tolerance = 1e-6;
    int maxIterations = 1000;
};

struct ViewDef {
    bool window = true;
    int width = 0;  // 0 = fit to the domain
    int height = 800;
    double playback = 12.0;  // wall-clock seconds for the whole simulation
    std::string range = "auto";  // auto | dynamic | fixed
    double rangeMin = 0, rangeMax = 1;
    int colormap = 0;
    int contours = 10;
    bool mesh = false;
    std::string unit = "\xC2\xB0" "C";
    // 3D only
    int mode = 2;  // 0 surface, 1 clip, 2 octant, 3 slices
    int axis = 0;  // clip axis
    bool cutSet = false;
    double cut[3] = {0, 0, 0};
    double yaw = -60.0, pitch = 25.0, zoom = 1.0;  // degrees, degrees, factor
    double spin = 0.0;                             // degrees per second of playback
};

struct OutputDef {
    std::string directory;
    double interval = 0.0;  // 0 = end / 120
    bool png = false, gif = true, vtk = false, csv = true;
    int width = 0, height = 600;
    double gifFps = 20.0;
    std::vector<ProbeDef> probes;
};

struct ProblemDef {
    std::string title;
    std::string configPath;
    MeshDef mesh;
    MaterialDef baseMaterial;
    std::vector<MaterialDef> materials;  // applied in order over the base material
    Expr initial;
    std::vector<BoundaryDef> boundaries;  // later definitions win on shared facets
    std::vector<SourceDef> sources;
    double faceH = 0.0;  // 2D plate model: convection from the large faces [W/(m^2 K)]
    Expr faceAmbient;
    int faceCount = 2;
    bool hasExact = false;
    Expr exact;
    TimeDef time;
    SolverDef solver;
    ViewDef view;
    OutputDef output;
};

ProblemDef parseProblem(const Config& cfg);
const char* colormapName(int index);
const char* viewModeName(int mode);

}  // namespace fc
