DefineConstant[ h = {0.2, Min 0, Max 1, Name "Resolution" } ];

SetFactory("OpenCASCADE");

// Unit square [0,1] x [0,1]
Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {0, 1, 0, h};

Line(1) = {1, 2}; // bottom
Line(2) = {2, 3}; // right
Line(3) = {3, 4}; // top
Line(4) = {4, 1}; // left

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Boundaries
Physical Curve(9) = {1, 2, 4}; // bottom + right + left (free slip)
Physical Curve(1) = {3};       // top (free surface)

Physical Surface(1) = {1};

Mesh.MshFileVersion = 2.2;
