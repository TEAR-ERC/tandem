DefineConstant[ hf = {0.030, Min 0, Max 1000, Name "Fault resolution" } ];
DefineConstant[ h = {50.0, Min 0, Max 1000, Name "Far boundary resolution" } ];
DefineConstant[ D = {400, Min 0, Max 10000, Name "Domain size" } ];

w = D;
d = D;
d1 = 2;
d2 = 12;
d3 = 17;
d4 = 24;
Point(2) = {w, 0, 0, h};
Point(3) = {w, -d, 0, h};
Point(5) = {0, -d, 0, h};
Point(6) = {0, 0, 0, hf};
Point(7) = {0, -d4, 0, hf};
Point(8) = {0, -d3, 0, hf};
Point(9) = {0, -d2, 0, hf};
Point(10) = {0, -d1, 0, hf};
Line(2) = {6, 2};
Line(3) = {2, 3};
Line(4) = {3, 5};
Line(7) = {7, 5};
Line(8) = {8, 7};
Line(9) = {9, 8};
Line(10) = {10, 9};
Line(11) = {6, 10};
Curve Loop(1) = {11, 10, 9, 8, 7, -4, -3, -2};
Plane Surface(1) = {1};
Physical Curve(1) = {2, 4};
Physical Curve(3) = {8, 9, 10, 11};
Physical Curve(5) = {3, 7};
Physical Surface(1) = {1};
Mesh.MshFileVersion = 2.2;
