DefineConstant[ h = {0.8, Min 0, Max 1, Name "Resolution" } ];

SetFactory("OpenCASCADE");

// Corner points
Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {0, 1, 0, h};

// Outer edges
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Create the surface
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Physical groups for meshing and boundary conditions
Physical Curve(5) = {1, 2, 3, 4};
Physical Surface(1) = {1};

// Mesh format
Mesh.MshFileVersion = 2.2;
