SetFactory("OpenCASCADE");

// Corner points
Point(1) = {-1, -1, 0};
Point(2) = { 1, -1, 0};
Point(3) = { 1,  1, 0};
Point(4) = {-1,  1, 0};

// Interface points (x = 0)
Point(5) = {0, -1, 0};
Point(6) = {0,  1, 0};

// Outer boundary lines
Line(1) = {1, 2}; // bottom
Line(2) = {2, 3}; // right
Line(3) = {3, 4}; // top
Line(4) = {4, 1}; // left

// Interface line
Line(5) = {5, 6};

// Split bottom/top edges
Line(6) = {1, 5}; // bottom-left
Line(7) = {5, 2}; // bottom-right
Line(8) = {3, 6}; // top-right
Line(9) = {6, 4}; // top-left

// Left surface (x <= 0)
Line Loop(1) = {6, 5, 9, 4};
Plane Surface(1) = {1};

// Right surface (x >= 0)
Line Loop(2) = {7, 2, 8, -5};
Plane Surface(2) = {2};

// Mesh refinement

Field[1] = Distance;
Field[1].EdgesList = {5}; // distance from interface

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.5;  // fine near interface

Background Field = 2;

// Tags

// Surfaces ("Volume tags")
Physical Surface(1) = {1}; // left
Physical Surface(2) = {2}; // right

// Boundaries
Physical Curve(5) = {2, 4};          // right + left boundaries (Dirichlet)
Physical Curve(1) = {6, 7, 8, 9};    // bottom + top boundaries (Natural)

Mesh.MshFileVersion = 2.2;
