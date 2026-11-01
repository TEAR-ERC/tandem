DefineConstant[ h = {0.25, Min 0, Max 1, Name "Resolution" } ];

// Unit box cut from side to side by a horizontal fault at y = 0.5, see fault_shear.lua.
Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 0.5, 0, h};
Point(4) = {1, 1, 0, h};
Point(5) = {0, 1, 0, h};
Point(6) = {0, 0.5, 0, h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line(7) = {6, 3};

Curve Loop(1) = {1, 2, -7, 6};
Plane Surface(1) = {1};
Curve Loop(2) = {7, 3, 4, 5};
Plane Surface(2) = {2};

Physical Curve(3) = {7};
Physical Curve(5) = {1, 2, 3, 4, 5, 6};
Physical Surface(1) = {1, 2};

Mesh.MshFileVersion = 2.2;
