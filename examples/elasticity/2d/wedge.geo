DefineConstant[ h = {0.020, Min 0, Max 10, Name "Resolution" } ];
DefineConstant[ dip = {60, Min 0, Max 90, Name "Dipping angle" } ];
DefineConstant[ L = {1.0, Min 0, Max 10, Name "Resolution" } ];

dip_rad = -dip * Pi / 180.0;

Point(1) = {L, 0, 0, h};
Point(2) = {0, 0, 0, h};
Point(3) = {L, L * Tan(dip_rad), 0, h};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};
Curve Loop(1) = {1, 2, 3};
Plane Surface(1) = {1};
Physical Curve(1) = {1};
Physical Curve(5) = {2, 3};
Physical Surface(1) = {1};
Mesh.MshFileVersion = 2.2;
