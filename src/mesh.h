#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace fc {

// Simplex mesh with linear (P1) elements: triangles in 2D, tetrahedra in 3D.
// The generators below make rectangles and boxes, but nothing downstream
// assumes structure, so other mesh sources (e.g. a Gmsh reader) only need to
// fill this struct.
struct Mesh {
    struct Face {       // boundary facet: an edge (2D) or a triangle (3D)
        uint32_t n[3];  // the first `dim` entries are used
        int side;       // index into sideNames
    };

    int dim = 2;
    std::vector<double> x, y, z;   // node coordinates (z = 0 in 2D)
    std::vector<uint32_t> cells;   // dim+1 nodes per cell, positively oriented
    std::vector<Face> boundary;
    std::vector<std::string> sideNames;                    // canonical names
    std::vector<std::pair<std::string, int>> sideAliases;  // e.g. xmin -> left
    std::string description;
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0, zmin = 0, zmax = 0;
    bool boxShaped = false;  // the domain is exactly its bounding box (rectangle / box)

    int nodesPerCell() const { return dim + 1; }
    int nodesPerFace() const { return dim; }
    size_t numNodes() const { return x.size(); }
    size_t numCells() const { return cells.size() / static_cast<size_t>(nodesPerCell()); }
    int sideIndex(const std::string& name) const;  // canonical name or alias; -1 if unknown
    std::string sideList() const;                  // "left right ... (aliases: xmin ...)"

    // Unique edges for the wireframe: every element edge in 2D, the edges of
    // the boundary triangles in 3D. Pairs of node indices.
    std::vector<uint32_t> wireframeEdges() const;
    // Boundary facets as a flat index list (2 per edge in 2D, 3 per triangle in 3D).
    std::vector<uint32_t> boundaryIndices() const;
};

enum class DiagonalPattern { Alternating, Right, Left };

// Rectangle [x0, x0+sx] x [y0, y0+sy] split into nx*ny cells, each cut into two
// triangles. Sides: left/right (x), bottom/top (y); aliases xmin..ymax.
Mesh makeRectangleMesh(double x0, double y0, double sx, double sy, int nx, int ny, DiagonalPattern pattern);

// Box [x0, x0+sx] x [y0, y0+sy] x [z0, z0+sz] split into nx*ny*nz cubes of six
// tetrahedra each (Kuhn/Freudenthal subdivision, conforming without having to
// alternate orientations). Sides: left/right (x), front/back (y), bottom/top
// (z); aliases xmin..zmax.
Mesh makeBoxMesh(double x0, double y0, double z0, double sx, double sy, double sz, int nx, int ny, int nz);

// A tube swept along a circular arc in the x-z plane (or a straight line along
// x when bend = 0), centred on `origin`. The radius varies along the tube, and
// the cross-section can be any star shape. Built by mapping a box mesh: the
// square cross-section goes onto the unit disc with the concentric
// (Shirley-Chiu) map, is scaled by radius(s) * section(angle, s) and is placed
// in the plane normal to the centre line. Angle 0 points along N (towards the
// centre of the bend; +z for a straight tube), +pi/2 along T x N (-y).
//
// With a cavity the tube is hollow: a round hole of radius cavity(s) runs
// along the centre line and is open at both ends. The box is then mapped onto
// the ring between the hole and the outer skin (through the wall x around,
// with the seam joined), keeping 4 * cellsAcross cells around like the solid
// tube's outer ring and cellsAcross / 2 (at least 2) through the wall.
//
// Sides: start (s = 0), end (s = 1), surface (the outer skin) and, when
// hollow, cavity (the wall of the hole).
struct TubeSpec {
    double origin[3] = {0, 0, 0};
    double length = 0.1;  // along the centre line [m]
    double bend = 0.0;    // radius of the centre-line arc [m]; 0 = straight
    std::function<double(double)> radius;           // radius(s), s in [0, 1] along the tube [m]
    std::function<double(double, double)> section;  // relative radius vs. (angle, s)
    std::function<double(double)> cavity;           // hole radius(s) [m]; empty = solid
    int cellsAlong = 64, cellsAcross = 12;
};
Mesh makeTubeMesh(const TubeSpec& spec);

}  // namespace fc
