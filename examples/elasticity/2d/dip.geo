DefineConstant[ h = {0.1, Min 0, Max 10, Name "Mesh size" } ];
DefineConstant[ dip = {60, Min 0, Max 90, Name "Dipping angle" } ];
DefineConstant[ R = {1.0, Min 0, Max 10, Name "H" } ];

dip_rad = -dip * Pi / 180.0;
Point(1) = {-R, 0, 0, h};
Point(2) = {0, 0, 0, h};
Point(3) = {R, 0, 0, h};
Point(4) = {R * Cos(dip_rad), R * Sin(dip_rad), 0, h};
Line(1) = {1, 2};
Line(2) = {3, 2};
Line(3) = {2, 4};
Circle(4) = {4, 2, 1};
Circle(5) = {4, 2, 3};
Curve Loop(1) = {1, 3, 4};
Curve Loop(2) = {2, 5, 3};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Physical Curve(1) = {1, 2};
Physical Curve(3) = {3};
Physical Curve(5) = {4, 5};
Physical Surface(1) = {1, 2};
Mesh.MshFileVersion = 2.2;
