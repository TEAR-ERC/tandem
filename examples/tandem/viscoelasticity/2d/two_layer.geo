// Two-layer box for two_layer.lua: the unit square split at y = 0.5, with the mesh
// conforming to the interface so that each element holds one material.
DefineConstant[ h = {0.25, Min 0, Max 1, Name "Resolution" } ];

Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 0.5, 0, h};
Point(4) = {0, 0.5, 0, h};
Point(5) = {1, 1, 0, h};
Point(6) = {0, 1, 0, h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {3, 5};
Line(6) = {5, 6};
Line(7) = {6, 4};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Curve Loop(2) = {-3, 5, 6, 7};
Plane Surface(2) = {2};

// Bottom: u = 0. Sides and top: prescribed traction (uniform shear stress).
Physical Curve(5) = {1};
Physical Curve(7) = {2, 4, 5, 6, 7};
Physical Surface(1) = {1, 2};

Mesh.MshFileVersion = 2.2;
