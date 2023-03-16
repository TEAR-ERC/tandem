DefineConstant[ hd = {0.05, Min 0, Max 1000, Name "Damage zone resolution" } ];
DefineConstant[ hf = {0.05, Min 0, Max 1000, Name "Fault resolution" } ];
DefineConstant[ h = {50.0, Min 0, Max 1000, Name "Far boundary resolution" } ];
DefineConstant[ D = {400, Min 0, Max 10000, Name "Domain size" } ];

w = D;
d = D;
d1 = 2;
d2 = 12;
d3 = 17;
d4 = 24;
fzw = 0.5;
fzd = 10;

Point(1) = {0, -d, 0, h};
Point(2) = {0, -d4, 0, hf};
Point(3) = {0, -d3, 0, hf};
Point(4) = {0, -d2, 0, hf};
Point(5) = {0, -d1, 0, hd};
Point(6) = {0, 0, 0, hd};
Point(7) = {w, 0, 0, h};
Point(8) = {w, -d, 0, h};

Point(21) = {fzw, 0, 0, hf};
Point(22) = {0, -fzd+fzw, 0, hd};
Point(23) = {fzw, -fzd+fzw, 0, hf};
Point(24) = {0, -fzd, 0, hd};

Line(1) = {6, 21};
Line(2) = {21, 7};
Line(3) = {7, 8};
Line(4) = {8, 1};
Line(5) = {1, 2};

Line(11) = {2, 3};
Line(12) = {3, 4};
Line(13) = {4, 24};
Line(14) = {24, 22};
Line(15) = {22, 5};
Line(16) = {5, 6};

Line(21) = {21, 23};
Circle(22) = {23, 22, 24};

Curve Loop(1) = {1, 21, 22, 14, 15, 16};
Curve Loop(2) = {2, 3, 4, 5, 11, 12, 13, -21, -22};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Physical Curve(1) = {1, 2, 4};
Physical Curve(3) = {12, 13, 14, 15, 16, 17};
Physical Curve(5) = {3, 5};
Physical Surface(1) = {1};
Physical Surface(2) = {2};
Mesh.MshFileVersion = 2.2;
