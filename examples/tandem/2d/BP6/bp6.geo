DefineConstant[ hf = {0.010, Min 0, Max 1000, Name "Fault resolution" } ];
DefineConstant[ h = {50.0, Min 0, Max 1000, Name "Far boundary resolution" } ];
DefineConstant[ D = {400, Min 0, Max 10000, Name "Domain size" } ];

w = D;
d = D/2;
lf = 20;
Point(1) = {0, d, 0, h};
Point(2) = {w, d, 0, h};
Point(3) = {w, -d, 0, h};
Point(4) = {0, -d, 0, h};
Point(11) = {0, lf, 0, hf};
Point(12) = {0, -lf, 0, hf};
Line(12) = {1, 2};
Line(23) = {2, 3};
Line(34) = {3, 4};
Line(111) = {1, 11};
Line(1112) = {11, 12};
Line(124) = {12, 4};
Curve Loop(1) = {111,1112,124,-34,-23,-12};
Plane Surface(1) = {1};
Physical Curve(3) = {1112};        // Fault 
Physical Curve(5) = {111,124};     // Dirichlet
Physical Surface(1) = {1};
Mesh.MshFileVersion = 2.2;